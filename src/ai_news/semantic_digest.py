"""
Semantic Digest Generator using FastEmbed

Generates topic digests using semantic embeddings instead of keyword matching.
"""

from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import numpy as np


class SemanticDigestGenerator:
    """Generate topic digests using semantic similarity."""

    def __init__(self, database, min_similarity: float = 0.58):
        """Initialize the semantic digest generator.

        Args:
            database: Database instance
            min_similarity: Minimum similarity threshold (0-1)
        """
        self.db = database
        self.min_similarity = min_similarity
        self.embedding_model = None

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

        # Combine topics into single query
        topic_query = " ".join(topics)
        topic_embedding = list(model.embed([topic_query]))[0]

        # Calculate similarities
        scored_articles = []
        for article, emb in zip(dated_articles, article_embeddings):
            similarity = np.dot(topic_embedding, emb) / (
                np.linalg.norm(topic_embedding) * np.linalg.norm(emb)
            )

            if similarity >= self.min_similarity:
                scored_articles.append((article, similarity))

        # Sort by similarity
        scored_articles.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        results = scored_articles[:top_k]

        return {
            'topics': topics,
            'articles': results,  # List of (article, similarity)
            'total': len(results),
            'method': 'semantic_fastembed',
            'threshold': self.min_similarity,
            'avg_similarity': sum(s for _, s in results) / len(results) if results else 0
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
            for i, (article, similarity) in enumerate(articles, 1):
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
