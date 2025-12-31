# src/ai_news/adaptive_threshold.py
"""Adaptive confidence threshold calculation based on article quality."""

import logging
from typing import List, Dict
from statistics import mean, stdev

logger = logging.getLogger(__name__)


def analyze_article_quality(articles: List) -> Dict:
    """
    Analyze collected articles to determine quality.

    Returns:
        dict with keys:
        - avg_confidence: float
        - variance: float (or 0 if only 1 article)
        - needs_higher_threshold: bool
    """
    if not articles:
        return {'avg_confidence': 0, 'variance': 0, 'needs_higher_threshold': True}

    # Get confidence scores (prefer intersection_confidence, fallback to ai_confidence)
    confidences = []
    for article in articles:
        # First try to extract intersection_confidence from ai_keywords_found
        conf = 0.3  # Default low confidence
        keywords = article.ai_keywords_found or []
        for kw in keywords:
            if isinstance(kw, str) and kw.startswith('intersection_confidence:'):
                try:
                    conf = float(kw.split(':')[1])
                    break
                except (ValueError, IndexError):
                    pass
        
        # Fallback to ai_confidence if no intersection_confidence found
        if conf == 0.3:
            conf = getattr(article, 'ai_confidence', 0.3)
        
        confidences.append(conf)

    avg_conf = mean(confidences)

    # Calculate variance
    if len(confidences) > 1:
        try:
            variance = stdev(confidences)
        except Exception:
            variance = 0
    else:
        variance = 0

    # Quality rules
    needs_higher = (
        avg_conf < 0.4 or  # Low average confidence
        variance < 0.15    # Low variance = topic too broad
    )

    return {
        'avg_confidence': avg_conf,
        'variance': variance,
        'needs_higher_threshold': needs_higher
    }


def calculate_adaptive_threshold(topic: str, articles: List, initial_threshold: float = 0.3) -> float:
    """
    Calculate adaptive threshold based on article quality.

    Args:
        topic: Topic name (for future domain-based logic)
        articles: List of collected Article objects
        initial_threshold: Starting threshold (default 0.3)

    Returns:
        float: Final threshold to use
    """
    quality = analyze_article_quality(articles)

    # If no articles or very low quality, use higher threshold
    if not articles or quality['avg_confidence'] < 0.35:
        return 0.6

    # If quality is moderate, bump to 0.5
    if quality['needs_higher_threshold']:
        return 0.5

    # Good quality, keep initial threshold
    return initial_threshold
