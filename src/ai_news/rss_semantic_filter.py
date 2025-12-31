# src/ai_news/rss_semantic_filter.py
"""Semantic filtering for RSS articles based on topics."""

import logging
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)


def filter_rss_by_topic(database, topics: List[str], threshold: float = 0.5) -> List:
    """
    Filter RSS articles by topic using two-stage pipeline.

    Stage 1: Keyword pre-filter (fast, SQL-based)
    Stage 2: Semantic scoring (accurate, embedding-based)

    Args:
        database: Database instance
        topics: List of topic names
        threshold: Minimum semantic similarity (default 0.5)

    Returns:
        List of filtered Article objects
    """
    from .config import Config
    from .semantic_digest import SemanticDigestGenerator

    # Load config to get topic keywords
    config = Config.load(Path('config.json'))

    # Get keywords for all topics (case-insensitive lookup)
    all_keywords = set()
    normalized_topics = []
    for topic in topics:
        # Try exact match first, then case-insensitive
        if topic in config.topics:
            normalized_topics.append(topic)
            topic_keywords = config.get_topic_keywords(topic)
            all_keywords.update(kw.lower() for kw in topic_keywords)
        else:
            # Case-insensitive lookup
            for config_topic in config.topics.keys():
                if config_topic.lower() == topic.lower():
                    normalized_topics.append(config_topic)
                    topic_keywords = config.get_topic_keywords(config_topic)
                    all_keywords.update(kw.lower() for kw in topic_keywords)
                    break

    # Stage 1: Keyword pre-filter
    logger.info(f"Stage 1: Keyword pre-filter with {len(all_keywords)} keywords")
    candidates = _keyword_prefilter(database, list(all_keywords))

    if not candidates:
        logger.info("No candidates after keyword pre-filter")
        return []

    # Early exit: very few candidates, skip semantic scoring
    if len(candidates) < 5:
        logger.info(f"Only {len(candidates)} candidates, skipping semantic scoring")
        return candidates

    # Stage 2: Semantic scoring
    logger.info(f"Stage 2: Semantic scoring {len(candidates)} candidates with threshold {threshold}")
    filtered = _semantic_score_candidates(candidates, topics, threshold, database)

    logger.info(f"RSS filter: {len(candidates)} → {len(filtered)} articles")
    return filtered


def _keyword_prefilter(database, keywords: List[str]) -> List:
    """
    Fast SQL-based keyword filtering.

    Returns articles where title/content contains ANY keyword.
    """
    all_articles = database.get_articles(limit=10000)  # Recent articles

    candidates = []
    for article in all_articles:
        text = f"{article.title} {article.content or ''} {article.summary or ''}".lower()

        # OR logic: match ANY keyword
        if any(kw in text for kw in keywords):
            candidates.append(article)

    return candidates


def _semantic_score_candidates(articles: List, topics: List[str], threshold: float, database) -> List:
    """
    Score articles using semantic embeddings with advanced scoring.

    Uses SemanticDigestGenerator's model for consistency, but scores only the
    provided candidates (not all database articles).

    Includes:
    - Title keyword boosting (+0.25)
    - Domain keyword boosting (+0.15)
    - Recency boosting
    - URL deduplication
    """
    try:
        import numpy as np
        from .semantic_digest import SemanticDigestGenerator
        from urllib.parse import urlparse, urlunparse
        from datetime import datetime

        generator = SemanticDigestGenerator(database, min_similarity=threshold)
        model = generator._get_model()

        # Prepare article texts for embedding
        article_texts = []
        for article in articles:
            text = f"{article.title}. {article.summary or article.content or ''}"
            article_texts.append(text[:2000])

        # Generate embeddings for candidates
        article_embeddings = list(model.embed(article_texts))

        # Extract domain keywords from topics for boosting
        domain_keywords = set()
        for topic in topics:
            words = topic.lower().split()
            for w in words:
                if len(w) > 3 and w not in {'artificial', 'intelligence', 'machine'}:
                    domain_keywords.add(w)

        # Generate embeddings for each topic
        topic_embeddings = {}
        for topic in topics:
            topic_query = f"AI {topic}" if 'ai' not in topic.lower() else topic
            topic_embedding = list(model.embed([topic_query]))[0]
            topic_embeddings[topic] = topic_embedding

        # Calculate similarity with boosting for each article against each topic
        scored_articles = []
        seen_urls = set()
        now = datetime.now()

        for article, article_emb in zip(articles, article_embeddings):
            # Find max similarity across all topics
            max_similarity = 0.0
            for topic, topic_emb in topic_embeddings.items():
                similarity = np.dot(topic_emb, article_emb) / (
                    np.linalg.norm(topic_emb) * np.linalg.norm(article_emb)
                )
                max_similarity = max(max_similarity, similarity)

            # Title keyword boost
            article_title_lower = article.title.lower() if article.title else ''
            has_title_keywords = any(
                kw in article_title_lower for kw in domain_keywords
            ) if domain_keywords else False
            title_boost = 0.25 if has_title_keywords else 0.0

            # Domain keyword boost
            article_text = f" {article.title} {article.summary or ''} {article.content or ''} ".lower()
            has_domain_relevance = any(
                kw in article_text for kw in domain_keywords
            ) if domain_keywords else False
            domain_boost = 0.15 if has_domain_relevance else 0.0

            # Recency boost
            recency_boost = 0.0
            if hasattr(article, 'published_at') and article.published_at:
                try:
                    days_old = (now - article.published_at.replace(tzinfo=None)).days
                    recency_boost = max(0, (7 - days_old) * 0.01)
                except:
                    pass

            # Apply boosts only for threshold check (not for logging)
            final_score = max_similarity + title_boost + domain_boost + recency_boost

            logger.debug(f"Article '{article.title[:50]}...': sim={max_similarity:.3f}, boosts={title_boost+domain_boost+recency_boost:.3f}, final={final_score:.3f}")

            # Check threshold with boosts
            if max_similarity >= threshold:
                # Normalize URL for deduplication
                try:
                    parsed = urlparse(article.url)
                    normalized_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
                except:
                    normalized_url = article.url

                if normalized_url not in seen_urls:
                    seen_urls.add(normalized_url)
                    scored_articles.append(article)
                    logger.debug(f"  -> Kept (similarity: {max_similarity:.3f})")

        return scored_articles

    except Exception as e:
        logger.error(f"Semantic scoring failed: {e}, returning keyword-filtered articles")
        return articles
