"""
Intersection Optimization for Multi-Domain Articles.

This module implements advanced semantic similarity scoring and
weighted intersection detection to boost multi-domain article
collection from 16.5% to 25%+ detection rate.

Phase 3 Implementation: Advanced intersection detection with
semantic awareness and context validation.
"""

import re
import math
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass  # Data may already exist

class IntersectionOptimizer:
    """
    Advanced intersection detection with semantic similarity
    and weighted scoring for multi-domain articles.
    """
    
    def __init__(self):
        """Initialize the intersection optimizer."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
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
        
        # Initialize spaCy for semantic similarity
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Warning: spaCy model not found, using simplified similarity")
            self.nlp = None
    
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
        
        # Use spaCy if available for better semantic understanding
        if self.nlp:
            try:
                doc1 = self.nlp(text1.lower().strip())
                doc2 = self.nlp(text2.lower().strip())
                return doc1.similarity(doc2)
            except:
                pass  # Fallback to simplified method
        
        # Fallback: Jaccard similarity with lemmatization
        return self._jaccard_similarity_lemmatized(text1, text2)
    
    def _jaccard_similarity_lemmatized(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity with lemmatization.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity score
        """
        # Tokenize and lemmatize
        tokens1 = self._preprocess_text(text1)
        tokens2 = self._preprocess_text(text2)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Calculate Jaccard similarity
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for similarity analysis.
        
        Args:
            text: Text to preprocess
            
        Returns:
            List of processed tokens
        """
        # Lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stop words and punctuation, lemmatize
        processed_tokens = []
        for token in tokens:
            if (token.isalpha() and 
                token not in self.stop_words and 
                len(token) > 2):
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return processed_tokens
    
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
        weights = field_weights or self.field_weights
        
        results = {
            'total_score': 0.0,
            'field_scores': {},
            'semantic_matches': [],
            'context_matches': [],
            'intersection_detected': False,
            'confidence': 0.0
        }
        
        # Get article text fields
        title = article.get('title', '')
        content = article.get('content', '')
        summary = article.get('summary', '')
        
        # Analyze each field
        for field_name, field_text in [('title', title), ('content', content), ('summary', summary)]:
            if not field_text:
                continue
            
            field_score = 0.0
            field_weight = weights.get(field_name, 0.5)
            
            # Check for keyword intersections in this field
            field_analysis = self._analyze_field_intersections(
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
    
    def _analyze_field_intersections(
        self, 
        text: str, 
        keywords: List[str], 
        field_name: str
    ) -> Dict:
        """
        Analyze intersections in a specific text field.
        
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
                intersection = self._find_closest_intersection(
                    text, kw1, pos1, kw2, pos2
                )
                
                if intersection:
                    results['matches'].append(intersection)
                    results['score'] += intersection['score']
                    
                    # Add semantic similarity if above threshold
                    if intersection.get('semantic_similarity', 0) >= self.similarity_thresholds['moderate']:
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
    
    def _find_closest_intersection(
        self, 
        text: str, 
        kw1: str, 
        positions1: List[int], 
        kw2: str, 
        positions2: List[int]
    ) -> Optional[Dict]:
        """
        Find the closest intersection between two keywords.
        
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
                    
                    # Calculate semantic similarity
                    semantic_sim = self.calculate_semantic_similarity(
                        kw1 + ' ' + kw2, context
                    )
                    
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
        
        # Semantic similarity component
        similarity_score = semantic_similarity
        
        # Text length component (prefer not too short, not too long segments)
        text_length = len(text)
        optimal_length = 500
        length_score = 1 - abs(text_length - optimal_length) / optimal_length
        length_score = max(0, min(1, length_score))
        
        # Combined weighted score
        total_score = (
            0.4 * distance_score +      # 40% distance
            0.4 * similarity_score +    # 40% semantic similarity
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
        
        # Semantic matches indicate true connection
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
    optimizer = create_intersection_optimizer()
    
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
    print("Intersection test result:")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Detected: {result['intersection_detected']}")
    print(f"Total Score: {result['total_score']:.3f}")
    
    validation = optimizer.validate_intersection_relevance(result, test_article)
    print(f"\nIs Relevant: {validation['is_relevant']}")
    print(f"Relevance Score: {validation['relevance_score']:.3f}")
    print(f"Quality Indicators: {validation['quality_indicators']}")