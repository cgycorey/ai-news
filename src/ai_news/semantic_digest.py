"""
Semantic Digest Generator using FastEmbed

Generates topic digests using semantic embeddings instead of keyword matching.
"""

from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import numpy as np
from urllib.parse import urlparse, urlunparse
from collections import defaultdict


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
        # Get articles
        articles = self.db.get_articles(limit=5000, ai_only=ai_only)

        if not articles:
            return {
                'topics': topics,
                'articles': [],
                'total': 0,
                'method': 'semantic_fastembed',
                'error': 'No articles found'
            }

        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
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

        # Get model
        model = self._get_model()

        # Prepare texts
        article_texts = []
        for article in dated_articles:
            text = f"{article.title}. {article.summary or article.content or ''}"
            article_texts.append(text[:2000])

        # Generate embeddings
        article_embeddings = list(model.embed(article_texts))

        # Combine topics into single query with better phrasing
        # For cross-domain topics, add "AI" context if not present
        topic_query = " ".join(topics)
        if not any('ai' in t.lower() for t in topics):
            topic_query = f"AI {topic_query}"  # Add AI context for better matching

        topic_embedding = list(model.embed([topic_query]))[0]

        # Extract domain keywords from topics for filtering
        domain_keywords = set()
        filter_only_mode = False  # If True, require domain keywords

        # Check if topics contain domain-specific terms (not just AI terms)
        topic_text = ' '.join(topics).lower()
        domain_specific = any(term in topic_text for term in
            {'healthcare', 'medical', 'medicine', 'hospital',
             'education', 'teaching', 'school', 'university', 'student',
             'finance', 'banking', 'financial', 'investment',
             'manufacturing', 'retail', 'transportation', 'agriculture'})

        filter_only_mode = domain_specific

        for topic in topics:
            # Extract meaningful words from topics (but not "machine", "learning", etc)
            words = topic.lower().split()
            for w in words:
                if len(w) > 3 and w not in {'artificial', 'intelligence', 'machine'}:
                    domain_keywords.add(w)

        # Calculate similarities with date boost
        scored_articles = []
        now = datetime.now()
        seen_urls = set()  # Track normalized URLs for deduplication

        for article, emb in zip(dated_articles, article_embeddings):
            similarity = np.dot(topic_embedding, emb) / (
                np.linalg.norm(topic_embedding) * np.linalg.norm(emb)
            )

            # Hard filter: for domain-specific topics, article MUST contain domain keywords
            # Skip this for general AI topics
            article_text = f"{article.title} {article.summary or ''} {article.content or ''}".lower()
            if domain_keywords:
                # Check if article contains at least one domain keyword
                has_domain_relevance = any(kw in article_text for kw in domain_keywords)
                if not has_domain_relevance:
                    continue  # Skip articles that don't mention the domain

            # Date recency boost (articles from today get +0.07, decay over 7 days)
            if article.published_at:
                days_old = (now - article.published_at.replace(tzinfo=None)).days
                recency_boost = max(0, (7 - days_old) * 0.01)
            else:
                recency_boost = 0

            final_score = similarity + recency_boost

            if similarity >= self.min_similarity:
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
