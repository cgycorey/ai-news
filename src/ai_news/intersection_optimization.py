"""
Intersection Optimization for Multi-Domain Articles.

This module provides intersection detection using semantic embeddings.
Pure FastEmbed-based implementation - no NLTK dependency.
"""

import re
import math
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)

# Import semantic intersection (primary implementation)
try:
    from .semantic_intersection import SemanticIntersectionOptimizer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logger.error("Semantic intersection not available. Install fastembed.")


class IntersectionOptimizer:
    """
    Advanced intersection detection using semantic embeddings.
    
    Pure FastEmbed-based implementation for multi-domain article detection.
    No NLTK or tokenization required.
    """
    
    def __init__(self):
        """Initialize the intersection optimizer."""
        self._semantic_optimizer = None
        
        # Initialize semantic optimizer
        if SEMANTIC_AVAILABLE:
            try:
                self._semantic_optimizer = SemanticIntersectionOptimizer()
                logger.info("Using semantic embeddings for intersection detection")
            except Exception as e:
                logger.error(f"Failed to initialize semantic optimizer: {e}")
                raise
        else:
            raise RuntimeError("Semantic intersection not available. Install fastembed.")
        
        # Field weights for intersection detection
        self.field_weights = {
            'title': 1.0,      # Title has highest relevance
            'content': 0.8,    # Content has high relevance
            'summary': 0.6     # Summary has moderate relevance
        }
        
        # Semantic similarity thresholds
        self.similarity_thresholds = {
            'strong': 0.9,     # Direct semantic match
            'moderate': 0.7,   # Related concepts
            'weak': 0.5        # Loosely related
        }
        
        # Context windows for local proximity
        self.context_window_size = 150  # characters
        self.sentence_window = 2        # sentences
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two text segments.
        
        Args:
            text1: First text segment
            text2: Second text segment
            
        Returns:
            Semantic similarity score (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0
        return self._semantic_optimizer.calculate_semantic_similarity(text1, text2)
    
    def detect_weighted_intersections(
        self, 
        article: Dict, 
        keywords: List[str], 
        field_weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Detect intersections with weighted scoring across fields.
        
        Args:
            article: Article dictionary with title, content, summary
            keywords: List of keywords to check for intersections
            field_weights: Custom field weights (optional)
        
        Returns:
            Dictionary with intersection analysis results
        """
        return self._semantic_optimizer.detect_semantic_intersections(
            article, keywords, field_weights
        )
    
    def validate_intersection_relevance(
        self, 
        intersection_data: Dict, 
        article: Dict
    ) -> Dict:
        """
        Validate if an intersection is truly relevant to the article.
        
        Args:
            intersection_data: Intersection analysis results
            article: Full article data
            
        Returns:
            Validation results with relevance scores
        """
        return self._semantic_optimizer.validate_intersection_relevance(
            intersection_data, article
        )


def create_intersection_optimizer() -> IntersectionOptimizer:
    """
    Factory function to create an intersection optimizer instance.
    
    Returns:
        Configured IntersectionOptimizer instance
    """
    return IntersectionOptimizer()


# Utility functions for intersection analysis

def analyze_intersection_improvement(
    before_results: Dict, 
    after_results: Dict
) -> Dict:
    """
    Analyze the improvement in intersection detection.
    
    Args:
        before_results: Results before optimization
        after_results: Results after optimization
        
    Returns:
        Improvement analysis
    """
    improvement = {
        'detection_rate_improvement': 0.0,
        'confidence_improvement': 0.0,
        'quality_improvement': 0.0,
        'summary': ''
    }
    
    before_rate = before_results.get('detection_rate', 0.0)
    after_rate = after_results.get('detection_rate', 0.0)
    
    improvement['detection_rate_improvement'] = after_rate - before_rate
    improvement['confidence_improvement'] = (
        after_results.get('avg_confidence', 0.0) - 
        before_results.get('avg_confidence', 0.0)
    )
    
    # Determine summary
    if improvement['detection_rate_improvement'] >= 0.08:  # 8% improvement
        improvement['summary'] = 'Excellent improvement'
    elif improvement['detection_rate_improvement'] >= 0.05:  # 5% improvement
        improvement['summary'] = 'Good improvement'
    elif improvement['detection_rate_improvement'] >= 0.02:  # 2% improvement
        improvement['summary'] = 'Modest improvement'
    else:
        improvement['summary'] = 'Minimal improvement'
    
    return improvement


if __name__ == '__main__':
    # Quick test of the intersection optimizer
    import sys
    import os
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    optimizer = create_intersection_optimizer()
    
    print(f"✅ IntersectionOptimizer created")
    print(f"Using semantic: True")
    print(f"Has semantic optimizer: {optimizer._semantic_optimizer is not None}")
    
    test_article = {
        'title': 'AI in Healthcare: Machine Learning Revolutionizes Medical Diagnosis',
        'content': """
        The integration of artificial intelligence and healthcare is transforming medical diagnosis.
        Machine learning algorithms are now being deployed in hospitals to detect diseases earlier
        and more accurately than traditional methods. This AI-powered healthcare revolution is
        particularly impactful in radiology and pathology, where machine learning models can
        analyze medical images with remarkable precision.
        """,
        'summary': 'AI and machine learning technologies are revolutionizing healthcare diagnosis through advanced algorithms.'
    }
    
    keywords = ['AI', 'Healthcare']
    
    result = optimizer.detect_weighted_intersections(test_article, keywords)
    print("\nIntersection test result:")
    print(f"Method: {result.get('method', 'unknown')}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Detected: {result['intersection_detected']}")
    print(f"Total Score: {result['total_score']:.3f}")
    
    validation = optimizer.validate_intersection_relevance(result, test_article)
    print(f"\nIs Relevant: {validation['is_relevant']}")
    print(f"Relevance Score: {validation['relevance_score']:.3f}")
    print(f"Quality Indicators: {validation['quality_indicators']}")