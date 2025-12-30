"""
Semantic Digest Generator using FastEmbed

Generates topic digests using semantic embeddings.
Uses the query itself as context - no hardcoded domain lists.
"""

from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import numpy as np
from urllib.parse import urlparse, urlunparse
from collections import defaultdict
import logging
import re

logger = logging.getLogger(__name__)

# Overly broad words that often cause false positives (soft warning, not exclusion)
BROAD_WARNING_WORDS = {
    'design': 'Consider using more specific terms like "fashion", "creative", "art"',
    'system': 'Consider specifying domain like "healthcare system", "education system"',
    'model': 'Consider specifying type like "LLM", "AI model", "ML model"',
    'framework': 'Consider specifying domain like "AI framework", "ML framework"',
}


class SemanticDigestGenerator:
    """Generate topic digests using semantic similarity."""

    def __init__(self, database, min_similarity: float = 0.62):
        """Initialize the semantic digest generator.

        Args:
            database: Database instance
            min_similarity: Minimum similarity threshold (0-1), default 0.62 for better quality
        """
        self.db = database
        self.min_similarity = min_similarity
        self.embedding_model = None

    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing tracking parameters."""
        try:
            parsed = urlparse(url)
            # Remove query parameters and fragment
            clean = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                '',  # Remove params
                '',  # Remove query
                ''   # Remove fragment
            ))
            return clean
        except Exception:
            return url

    def _get_model(self):
        """Lazy load FastEmbed model."""
        if self.embedding_model is None:
            from fastembed import TextEmbedding
            self.embedding_model = TextEmbedding()
        return self.embedding_model

    def generate_digest(
        self,
        topics: List[str],
        days: int = 7,
        ai_only: bool = True,
        top_k: int = 20
    ) -> Dict:
        """Generate a semantic topic digest.

        Args:
            topics: List of topics (can be single topic or multiple)
            days: Days to look back
            ai_only: Only AI-relevant articles
            top_k: Max articles per topic

        Returns:
            Dict with digest results
        """
        # Filter by date FIRST to reduce dataset
        cutoff_date = datetime.now() - timedelta(days=days)

        # Get articles with date filter in query (more efficient)
        # Reduce limit to avoid fetching too many
        articles = self.db.get_articles(limit=100, ai_only=ai_only)

        if not articles:
            return {
                'topics': topics,
                'articles': [],
                'total': 0,
                'method': 'semantic_fastembed',
                'error': 'No articles found'
            }

        # Filter by date (done in Python, but only on fetched results)
        dated_articles = [
            a for a in articles
            if a.published_at and a.published_at.replace(tzinfo=None) >= cutoff_date
        ]

        if not dated_articles:
            return {
                'topics': topics,
                'articles': [],
                'total': 0,
                'method': 'semantic_fastembed',
                'error': f'No articles in last {days} days'
            }

        # Pure semantic matching - no keyword filtering
        # FastEmbed handles topic matching automatically
        # Limit to 50 articles for performance (was 100)
        dated_articles = dated_articles[:50]
        logger.info(f"Using {len(dated_articles)} articles for semantic matching")

        # Get model
        model = self._get_model()

        # Prepare texts - only for filtered articles
        article_texts = []
        for article in dated_articles:
            text = f"{article.title}. {article.summary or article.content or ''}"
            article_texts.append(text[:2000])

        # Generate embeddings - only for filtered articles
        article_embeddings = list(model.embed(article_texts))

        # Combine topics into single query with better phrasing
        # Default: prepend "AI" since this is AI news
        topic_query = " ".join(topics)
        if not any('ai' in t.lower() for t in topics):
            topic_query = f"AI {topic_query}"

        topic_embedding = list(model.embed([topic_query]))[0]

        # Use fixed threshold for all topics
        effective_threshold = self.min_similarity

        # Calculate similarities with date boost
        scored_articles = []
        now = datetime.now()
        seen_urls = set()  # Track normalized URLs for deduplication

        for article, emb in zip(dated_articles, article_embeddings):
            similarity = np.dot(topic_embedding, emb) / (
                np.linalg.norm(topic_embedding) * np.linalg.norm(emb)
            )

            # Date recency boost (articles from today get +0.07, decay over 7 days)
            if article.published_at:
                days_old = (now - article.published_at.replace(tzinfo=None)).days
                recency_boost = max(0, (7 - days_old) * 0.01)
            else:
                recency_boost = 0

            final_score = similarity + recency_boost

            if similarity >= effective_threshold:
                # Normalize URL for deduplication
                normalized_url = self._normalize_url(article.url)
                if normalized_url not in seen_urls:
                    seen_urls.add(normalized_url)
                    scored_articles.append((article, similarity, final_score))

        # Sort by final_score (similarity + recency)
        scored_articles.sort(key=lambda x: x[2], reverse=True)

        # Apply source diversity: max 2 articles per source in top results
        diversified_results = []
        source_counts = defaultdict(int)
        for article, similarity, final_score in scored_articles:
            source = article.source_name
            if source_counts[source] < 2:  # Max 2 per source
                diversified_results.append((article, similarity, final_score))
                source_counts[source] += 1

            if len(diversified_results) >= top_k:
                break

        results = diversified_results

        return {
            'topics': topics,
            'articles': results,  # List of (article, similarity, final_score)
            'total': len(results),
            'method': 'semantic_fastembed',
            'threshold': self.min_similarity,
            'avg_similarity': sum(s for _, s, _ in results) / len(results) if results else 0
        }

    def format_markdown(self, digest_result: Dict) -> str:
        """Format digest result as markdown.

        Args:
            digest_result: Result from generate_digest()

        Returns:
            Markdown formatted digest
        """
        topics = digest_result['topics']
        articles = digest_result['articles']

        topics_str = ', '.join(topics)

        md = f"""# Topic Analysis: {topics_str}
*Generated using semantic embeddings (FastEmbed)*

## 📈 Overview

- **Method:** Semantic matching (no predefined keywords)
- **Threshold:** {digest_result.get('threshold', 0.58)}
- **Total Articles:** {digest_result['total']}
"""

        if 'avg_similarity' in digest_result:
            avg_pct = int(digest_result['avg_similarity'] * 100)
            md += f"- **Avg Similarity:** {avg_pct}%\n"

        md += "\n## 📰 Articles\n\n"

        if not articles:
            md += f"No articles found for '{topics_str}' with semantic similarity >= {digest_result.get('threshold', 0.58)}\n"
        else:
            for i, (article, similarity, final_score) in enumerate(articles, 1):
                match_pct = int(similarity * 100)

                md += f"### {i}. 🤖 {article.title}\n\n"
                md += f"**Source:** {article.source_name}\n"

                if article.published_at:
                    md += f"**Date:** {article.published_at.strftime('%Y-%m-%d')}\n"

                md += f"**Similarity:** {match_pct}%\n"
                md += f"**Category:** {article.category}\n\n"

                if article.summary:
                    md += f"{article.summary}\n\n"

                md += f"**Read more:** [{article.url}]({article.url})\n\n"

                if article.ai_keywords_found:
                    md += f"**AI Keywords:** {', '.join(article.ai_keywords_found[:5])}\n\n"

                md += "---\n\n"

        return md
