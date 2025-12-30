"""
Semantic Intersection Optimization using FastEmbed Embeddings.

Replaces NLTK-based intersection detection with pure semantic embeddings.
More accurate, faster, and no NLTK dependency.
"""

import re
import math
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SemanticIntersectionOptimizer:
    """
    Intersection detection using semantic embeddings.
    
    Uses FastEmbed to calculate semantic similarity between keywords
    and article text, detecting multi-domain intersections without NLTK.
    """
    
    def __init__(self, min_similarity: float = 0.55):
        """Initialize the semantic intersection optimizer.
        
        Args:
            min_similarity: Minimum similarity threshold for matches
        """
        self.min_similarity = min_similarity
        self.embedding_model = None
        
        # Field weights for intersection detection
        self.field_weights = {
            'title': 1.0,      # Title has highest relevance
            'content': 0.8,    # Content has high relevance
            'summary': 0.6     # Summary has moderate relevance
        }
        
        # Context window size for extracting text around keywords
        self.context_window_size = 150  # characters
    
    def _get_model(self):
        """Lazy load FastEmbed model."""
        if self.embedding_model is None:
            try:
                from fastembed import TextEmbedding
                self.embedding_model = TextEmbedding()
                logger.info("FastEmbed loaded for intersection optimization")
            except ImportError:
                logger.error("FastEmbed not available. Install with: pip install fastembed")
                raise
        return self.embedding_model
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using embeddings.
        
        Args:
            text1: First text segment
            text2: Second text segment
            
        Returns:
            Semantic similarity score (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        try:
            model = self._get_model()
            
            # Generate embeddings
            emb1 = list(model.embed([text1]))[0]
            emb2 = list(model.embed([text2]))[0]
            
            # Calculate cosine similarity
            import numpy as np
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity failed: {e}")
            return 0.0
    
    def detect_semantic_intersections(
        self,
        article: Dict,
        keywords: List[str],
        field_weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Detect intersections using semantic similarity.
        
        Instead of tokenization and Jaccard similarity, uses embeddings
        to find if keywords appear together in semantic context.
        
        Args:
            article: Article dictionary with title, content, summary
            keywords: List of keywords to check for intersections
            field_weights: Custom field weights (optional)
        
        Returns:
            Dictionary with intersection analysis results
        """
        weights = field_weights or self.field_weights
        
        results = {
            'total_score': 0.0,
            'field_scores': {},
            'semantic_matches': [],
            'context_matches': [],
            'intersection_detected': False,
            'confidence': 0.0,
            'method': 'semantic_embeddings'
        }
        
        # Get article text fields
        title = article.get('title', '')
        content = article.get('content', '')
        summary = article.get('summary', '')
        
        # Analyze each field
        for field_name, field_text in [('title', title), ('content', content), ('summary', summary)]:
            if not field_text:
                continue
            
            field_weight = weights.get(field_name, 0.5)
            
            # Check for keyword intersections in this field
            field_analysis = self._analyze_field_semantic_intersections(
                field_text, keywords, field_name
            )
            
            # Calculate weighted score
            field_score = field_analysis['score'] * field_weight
            results['field_scores'][field_name] = {
                'raw_score': field_analysis['score'],
                'weighted_score': field_score,
                'matches': field_analysis['matches']
            }
            
            results['total_score'] += field_score
            
            # Add semantic and context matches
            results['semantic_matches'].extend(field_analysis['semantic_matches'])
            results['context_matches'].extend(field_analysis['context_matches'])
        
        # Normalize total score (max possible is sum of weights)
        max_score = sum(weights.values())
        results['confidence'] = results['total_score'] / max_score if max_score > 0 else 0.0
        
        # Determine if intersection detected based on thresholds
        results['intersection_detected'] = results['confidence'] >= 0.25  # 25% threshold
        
        return results
    
    def _analyze_field_semantic_intersections(
        self,
        text: str,
        keywords: List[str],
        field_name: str
    ) -> Dict:
        """
        Analyze intersections in a specific text field using semantics.
        
        Args:
            text: Text content to analyze
            keywords: Keywords to find intersections for
            field_name: Name of the field being analyzed
            
        Returns:
            Analysis results for this field
        """
        results = {
            'score': 0.0,
            'matches': [],
            'semantic_matches': [],
            'context_matches': []
        }
        
        if len(keywords) < 2:
            return results
        
        # Find keyword positions
        keyword_positions = self._find_keyword_positions(text, keywords)
        
        # Check intersections between each keyword pair
        for i, kw1 in enumerate(keywords):
            for j, kw2 in enumerate(keywords):
                if i >= j:  # Skip duplicates and self-pairs
                    continue
                
                pos1 = keyword_positions.get(kw1, [])
                pos2 = keyword_positions.get(kw2, [])
                
                if not pos1 or not pos2:
                    continue
                
                # Find closest intersection
                intersection = self._find_closest_semantic_intersection(
                    text, kw1, pos1, kw2, pos2
                )
                
                if intersection:
                    results['matches'].append(intersection)
                    results['score'] += intersection['score']
                    
                    # Add semantic similarity if above threshold
                    if intersection.get('semantic_similarity', 0) >= self.min_similarity:
                        results['semantic_matches'].append(intersection)
                    
                    # Add context match if within distance
                    if intersection.get('within_context', False):
                        results['context_matches'].append(intersection)
        
        return results
    
    def _find_keyword_positions(self, text: str, keywords: List[str]) -> Dict[str, List[int]]:
        """
        Find all positions of each keyword in the text.
        
        Args:
            text: Text to search
            keywords: Keywords to find
            
        Returns:
            Dictionary mapping keyword to list of start positions
        """
        positions = {}
        text_lower = text.lower()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            keyword_positions = []
            
            # Find all occurrences of the keyword
            start = 0
            while True:
                pos = text_lower.find(keyword_lower, start)
                if pos == -1:
                    break
                keyword_positions.append(pos)
                start = pos + 1
            
            positions[keyword] = keyword_positions
        
        return positions
    
    def _find_closest_semantic_intersection(
        self,
        text: str,
        kw1: str,
        positions1: List[int],
        kw2: str,
        positions2: List[int]
    ) -> Optional[Dict]:
        """
        Find the closest semantic intersection between two keywords.
        
        Args:
            text: Full text
            kw1: First keyword
            positions1: Positions of first keyword
            kw2: Second keyword
            positions2: Positions of second keyword
            
        Returns:
            Intersection information or None if no good intersection found
        """
        best_intersection = None
        min_distance = float('inf')
        
        for pos1 in positions1:
            for pos2 in positions2:
                distance = abs(pos1 - pos2)
                
                if distance < min_distance:
                    min_distance = distance
                    
                    # Extract context around intersection
                    start_pos = max(0, min(pos1, pos2) - self.context_window_size)
                    end_pos = min(len(text), max(pos1, pos2) + len(kw1) + len(kw2) + self.context_window_size)
                    context = text[start_pos:end_pos]
                    
                    # Calculate semantic similarity between keywords and context
                    # This checks if the context semantically relates to BOTH keywords
                    kw_context = f"{kw1} {kw2}"
                    semantic_sim = self.calculate_semantic_similarity(kw_context, context)
                    
                    # Calculate intersection score
                    score = self._calculate_intersection_score(
                        distance, semantic_sim, text
                    )
                    
                    best_intersection = {
                        'keyword1': kw1,
                        'keyword2': kw2,
                        'distance': distance,
                        'context': context,
                        'semantic_similarity': semantic_sim,
                        'score': score,
                        'within_context': distance <= self.context_window_size,
                        'text_segment': self._extract_text_segment(text, pos1, pos2)
                    }
        
        return best_intersection
    
    def _calculate_intersection_score(
        self,
        distance: int,
        semantic_similarity: float,
        text: str
    ) -> float:
        """
        Calculate score for an intersection based on distance and similarity.
        
        Args:
            distance: Distance between keywords
            semantic_similarity: Semantic similarity score
            text: Full text for additional context
            
        Returns:
            Intersection score (0.0 to 1.0)
        """
        # Distance component (closer is better)
        max_distance = 500  # Maximum distance to consider
        distance_score = max(0, 1 - (distance / max_distance))
        
        # Semantic similarity component (using embeddings)
        similarity_score = semantic_similarity
        
        # Text length component (prefer not too short, not too long segments)
        text_length = len(text)
        optimal_length = 500
        length_score = 1 - abs(text_length - optimal_length) / optimal_length
        length_score = max(0, min(1, length_score))
        
        # Combined weighted score
        total_score = (
            0.4 * distance_score +      # 40% distance
            0.4 * similarity_score +    # 40% semantic similarity (embeddings)
            0.2 * length_score          # 20% text length appropriateness
        )
        
        return min(1.0, total_score)
    
    def _extract_text_segment(
        self,
        text: str,
        pos1: int,
        pos2: int
    ) -> str:
        """
        Extract a text segment around the intersection point.
        
        Args:
            text: Full text
            pos1: First keyword position
            pos2: Second keyword position
            
        Returns:
            Text segment around the intersection
        """
        start = max(0, min(pos1, pos2) - 50)
        end = min(len(text), max(pos1, pos2) + 50)
        
        segment = text[start:end]
        # Add ellipsis if truncated
        if start > 0:
            segment = '...' + segment
        if end < len(text):
            segment = segment + '...'
        
        return segment
    
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
        validation = {
            'is_relevant': False,
            'relevance_score': 0.0,
            'quality_indicators': [],
            'warnings': []
        }
        
        if not intersection_data or not intersection_data.get('intersection_detected', False):
            return validation
        
        # Check context matches (strong indicator)
        context_matches = intersection_data.get('context_matches', [])
        semantic_matches = intersection_data.get('semantic_matches', [])
        
        relevance_score = 0.0
        
        # Context matches are very valuable
        if context_matches:
            relevance_score += 0.4
            validation['quality_indicators'].append('context_proximity')
        
        # Semantic matches indicate true connection (using embeddings)
        if semantic_matches:
            relevance_score += 0.3
            validation['quality_indicators'].append('semantic_relevance')
        
        # Check overall confidence
        confidence = intersection_data.get('confidence', 0.0)
        relevance_score += confidence * 0.2
        
        # Check field distribution (good intersections appear in multiple fields)
        field_scores = intersection_data.get('field_scores', {})
        fields_with_matches = sum(1 for fs in field_scores.values() if fs.get('matches'))
        if fields_with_matches >= 2:
            relevance_score += 0.1
            validation['quality_indicators'].append('multi_field_presence')
        
        # Normalization and threshold
        validation['relevance_score'] = min(1.0, relevance_score)
        validation['is_relevant'] = validation['relevance_score'] >= 0.5
        
        # Add warnings for potentially weak intersections
        if confidence < 0.3:
            validation['warnings'].append('low_confidence')
        if not context_matches and not semantic_matches:
            validation['warnings'].append('no_semantic_context')
        
        return validation


def create_semantic_intersection_optimizer(min_similarity: float = 0.55) -> SemanticIntersectionOptimizer:
    """
    Factory function to create a semantic intersection optimizer instance.
    
    Args:
        min_similarity: Minimum similarity threshold
        
    Returns:
        Configured SemanticIntersectionOptimizer instance
    """
    return SemanticIntersectionOptimizer(min_similarity=min_similarity)


# Backward compatibility alias
IntersectionOptimizer = SemanticIntersectionOptimizer
create_intersection_optimizer = create_semantic_intersection_optimizer