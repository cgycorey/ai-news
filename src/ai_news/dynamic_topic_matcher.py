"""
Dynamic Topic Matcher using FastEmbed

Uses semantic embeddings to find articles related to a topic,
without requiring predefined keyword lists.
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict


class DynamicTopicMatcher:
    """Find articles related to topics using semantic embeddings."""

    def __init__(self, database, min_similarity: float = 0.60):
        """Initialize the matcher.

        Args:
            database: Database instance
            min_similarity: Minimum similarity threshold (0-1)
        """
        self.db = database
        self.min_similarity = min_similarity
        self.embedding_model = None
        self._cache = {}

    def _get_embedding_model(self):
        """Lazy load FastEmbed model."""
        if self.embedding_model is None:
            try:
                from fastembed import TextEmbedding
                self.embedding_model = TextEmbedding()
                print("✅ FastEmbed model loaded")
            except ImportError:
                print("⚠️  FastEmbed not available, install with: pip install fastembed")
                raise
        return self.embedding_model

    def find_related_articles(
        self,
        topic: str,
        days: int = 30,
        ai_only: bool = True,
        top_k: int = 20
    ) -> List[Tuple[object, float]]:
        """Find articles semantically related to a topic.

        Args:
            topic: Topic to search for
            days: Days to look back
            ai_only: Only search AI-relevant articles
            top_k: Max number of results

        Returns:
            List of (article, similarity_score) tuples
        """
        # Get articles
        articles = self.db.get_articles(limit=5000, ai_only=ai_only)

        if not articles:
            return []

        # Create embeddings
        model = self._get_embedding_model()

        # Embed topic
        topic_embedding = list(model.embed([topic]))[0]

        # Prepare article texts
        article_texts = []
        for article in articles:
            text = f"{article.title}. {article.summary or article.content or ''}"
            # Truncate to avoid issues
            article_texts.append(text[:2000])

        # Embed all articles
        article_embeddings = list(model.embed(article_texts))

        # Calculate similarities
        results = []
        for i, (article, emb) in enumerate(zip(articles, article_embeddings)):
            # Cosine similarity
            similarity = np.dot(topic_embedding, emb) / (
                np.linalg.norm(topic_embedding) * np.linalg.norm(emb)
            )

            if similarity >= self.min_similarity:
                results.append((article, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def expand_topic_semantic(
        self,
        topic: str,
        days: int = 30,
        ai_only: bool = True,
        top_k: int = 10
    ) -> Dict:
        """Expand a topic by finding semantically related articles.

        Args:
            topic: Base topic
            days: Days to look back
            ai_only: Only AI-relevant articles
            top_k: Max results

        Returns:
            Dict with:
                - 'topic': original topic
                - 'matches': list of (article, similarity)
                - 'count': number of matches
                - 'avg_similarity': average similarity
        """
        results = self.find_related_articles(topic, days, ai_only, top_k)

        if not results:
            return {
                'topic': topic,
                'matches': [],
                'count': 0,
                'avg_similarity': 0.0
            }

        avg_sim = sum(s for _, s in results) / len(results)

        return {
            'topic': topic,
            'matches': results,
            'count': len(results),
            'avg_similarity': avg_sim
        }

    def find_similar_topics(
        self,
        topic: str,
        candidate_topics: List[str]
    ) -> List[Tuple[str, float]]:
        """Find topics semantically similar to the given topic.

        Args:
            topic: Base topic
            candidate_topics: List of candidate topics

        Returns:
            List of (topic, similarity) sorted by similarity
        """
        model = self._get_embedding_model()

        # Embed base topic
        topic_emb = list(model.embed([topic]))[0]

        # Embed candidates
        candidate_embs = list(model.embed(candidate_topics))

        # Calculate similarities
        similarities = []
        for candidate, emb in zip(candidate_topics, candidate_embs):
            sim = np.dot(topic_emb, emb) / (
                np.linalg.norm(topic_emb) * np.linalg.norm(emb)
            )
            similarities.append((candidate, sim))

        # Sort
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities
