#!/usr/bin/env python3
"""
Enhanced Multi-Keyword Collector for AI News

This module provides enhanced multi-keyword combination support addressing:
- Multi-keyword scoring and ranking
- Keyword intersection detection
- Region-specific keyword optimization
- Performance optimization for large keyword sets
- Advanced filtering combinations (AI + insurance + region)
"""

import re
import time
from typing import List, Dict, Tuple, Any, Set, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


@dataclass
class KeywordMatch:
    """Represents a keyword match with scoring information."""
    keyword: str
    category: str
    score: float
    position: int
    context: str
    weight: float = 1.0


@dataclass
class MultiKeywordResult:
    """Enhanced result for multi-keyword analysis."""
    is_relevant: bool
    total_score: float
    matches: List[KeywordMatch]
    category_scores: Dict[str, float]
    intersection_score: float
    region_boost: float
    final_score: float
    execution_time: float


@dataclass
class KeywordCategory:
    """Represents a category of keywords with weights."""
    name: str
    keywords: List[str]
    weight: float = 1.0
    variations: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.variations is None:
            self.variations = {}


class EnhancedMultiKeywordCollector:
    """Enhanced collector with multi-keyword scoring and intersection detection."""
    
    def __init__(self, performance_mode: bool = True):
        self.performance_mode = performance_mode
        self.keyword_index = {}  # For performance optimization
        self.regional_keywords = {}  # Region-specific keyword boosts
        
        # Default keyword variations for common AI terms
        self.default_variations = {
            'ai': ['ai', 'a.i.', 'artificial intelligence'],
            'ml': ['ml', 'machine learning'],
            'dl': ['dl', 'deep learning'],
            'nlp': ['nlp', 'natural language processing'],
            'llm': ['llm', 'large language model'],
            'gpt': ['gpt', 'chatgpt', 'gpt-3', 'gpt-4'],
            'api': ['api', 'application programming interface']
        }
        
        # Initialize keyword categories for common use cases
        self.init_default_categories()
    
    def init_default_categories(self):
        """Initialize default keyword categories."""
        self.categories = {
            'ai': KeywordCategory(
                name='ai',
                keywords=[
                    'AI', 'artificial intelligence', 'machine learning', 'deep learning',
                    'LLM', 'GPT', 'ChatGPT', 'neural network', 'algorithm', 'automation',
                    'algorithmic', 'transformer', 'BERT', 'NLP', 'computer vision'
                ],
                weight=1.0,
                variations=self.default_variations
            ),
            'insurance': KeywordCategory(
                name='insurance',
                keywords=[
                    'insurance', 'insurtech', 'underwriting', 'claims', 'risk', 
                    'premium', 'coverage', 'deductible', 'policy', 'actuarial',
                    'reinsurance', 'broker', ' Lloyd\'s'
                ],
                weight=0.8
            ),
            'healthcare': KeywordCategory(
                name='healthcare',
                keywords=[
                    'healthcare', 'medical', 'diagnostics', 'medicine', 'clinical',
                    'hospital', 'biotech', 'pharmaceutical', 'drug discovery',
                    'medical imaging', 'telemedicine', 'health'
                ],
                weight=0.8
            ),
            'fintech': KeywordCategory(
                name='fintech',
                keywords=[
                    'fintech', 'banking', 'financial', 'trading', 'payments',
                    'fraud detection', 'regtech', 'compliance', 'anti-money laundering',
                    'AML', 'digital banking'
                ],
                weight=0.8
            )
        }
        
        # Initialize regional keyword boosts
        self.regional_keywords = {
            'uk': {
                'boost_terms': ['London', 'British', 'UK', 'Lloyd\'s', 'NHS'],
                'boost_factor': 1.2
            },
            'us': {
                'boost_terms': ['American', 'US', 'FDA', 'HHS', 'Wall Street'],
                'boost_factor': 1.1
            },
            'eu': {
                'boost_terms': ['European', 'EU', 'Eurozone', 'GDPR'],
                'boost_factor': 1.1
            },
            'apac': {
                'boost_terms': ['Asia-Pacific', 'APAC', 'Singapore', 'Tokyo'],
                'boost_factor': 1.1
            }
        }
    
    def build_keyword_index(self, categories: Dict[str, KeywordCategory]):
        """Build performance index for keyword matching."""
        self.keyword_index = {}
        
        for category_name, category in categories.items():
            for keyword in category.keywords:
                # Index the base keyword
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append((category_name, category.weight))
                
                # Index variations
                if keyword.lower() in category.variations:
                    for variation in category.variations[keyword.lower()]:
                        if variation not in self.keyword_index:
                            self.keyword_index[variation] = []
                        self.keyword_index[variation].append((category_name, category.weight * 0.9))  # Slightly lower weight for variations
    
    def matches_word_boundary(self, keyword: str, text: str) -> List[Tuple[int, int]]:
        """Find all word boundary matches for a keyword in text."""
        positions = []
        
        # For patterns with dots, use more flexible matching
        if '.' in keyword:
            escaped_keyword = re.escape(keyword)
            pattern = rf'(?<!\w){escaped_keyword}(?!\w)'
        else:
            # Standard word boundary matching
            escaped_keyword = re.escape(keyword)
            pattern = rf'\b{escaped_keyword}\b'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            positions.append((match.start(), match.end()))
        
        return positions
    
    def fuzzy_match(self, keyword: str, text: str, threshold: float = 0.85) -> List[Tuple[int, int]]:
        """Find fuzzy matches for keyword in text."""
        matches = []
        words = text.split()
        
        for i, word in enumerate(words):
            similarity = SequenceMatcher(None, keyword.lower(), word.lower()).ratio()
            if similarity >= threshold:
                # Find approximate position in original text
                word_positions = [m.start() for m in re.finditer(rf'\b{re.escape(word)}\b', text, re.IGNORECASE)]
                for pos in word_positions:
                    matches.append((pos, pos + len(word)))
        
        return matches
    
    def extract_context(self, text: str, position: int, window: int = 50) -> str:
        """Extract context around a keyword match."""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end].strip()
    
    def calculate_intersection_score(self, category_matches: Dict[str, List[KeywordMatch]]) -> float:
        """Calculate intersection score for articles matching multiple categories."""
        categories_with_matches = [cat for cat, matches in category_matches.items() if matches]
        
        if len(categories_with_matches) < 2:
            return 0.0
        
        # Base intersection score increases with more categories
        base_score = min(len(categories_with_matches) * 0.3, 1.0)
        
        # Bonus for specific combinations
        high_value_combinations = [
            {'ai', 'insurance'},
            {'ai', 'healthcare'},
            {'ai', 'fintech'},
            {'insurance', 'healthcare'},
            {'fintech', 'insurance'}
        ]
        
        category_set = set(categories_with_matches)
        for combo in high_value_combinations:
            if combo.issubset(category_set):
                base_score += 0.2
        
        return min(base_score, 1.0)
    
    def calculate_region_boost(self, text: str, region: str) -> float:
        """Calculate region-specific boost for keywords."""
        if region not in self.regional_keywords:
            return 1.0
        
        region_config = self.regional_keywords[region]
        text_lower = text.lower()
        
        # Count region-specific terms
        region_terms_found = sum(1 for term in region_config['boost_terms'] if term.lower() in text_lower)
        
        if region_terms_found > 0:
            return region_config['boost_factor']
        
        return 1.0
    
    def analyze_multi_keywords(
        self, 
        title: str, 
        content: str, 
        categories: Dict[str, KeywordCategory] = None,
        region: str = "global",
        min_score: float = 0.1
    ) -> MultiKeywordResult:
        """
        Perform enhanced multi-keyword analysis with scoring.
        
        Args:
            title: Article title
            content: Article content  
            categories: Keyword categories to analyze
            region: Article region for regional boost
            min_score: Minimum score threshold
            
        Returns:
            MultiKeywordResult with comprehensive analysis
        """
        start_time = time.time()
        
        if categories is None:
            categories = self.categories
        
        # Build performance index if needed
        if self.performance_mode and not self.keyword_index:
            self.build_keyword_index(categories)
        
        # Combine title and content for analysis
        full_text = (title + " " + content).lower()
        
        # Analyze each category
        category_matches = {}
        category_scores = {}
        
        for category_name, category in categories.items():
            matches = []
            total_category_score = 0.0
            
            for keyword in category.keywords:
                keyword_lower = keyword.lower()
                
                # Direct word boundary matching
                positions = self.matches_word_boundary(keyword_lower, full_text)
                
                # Check variations if no direct matches
                if not positions and keyword_lower in category.variations:
                    for variation in category.variations[keyword_lower]:
                        positions.extend(self.matches_word_boundary(variation, full_text))
                
                # Fuzzy matching as fallback
                if not positions:
                    fuzzy_positions = self.fuzzy_match(keyword_lower, full_text, 0.85)
                    positions.extend(fuzzy_positions)
                
                # Create keyword matches
                for pos, end_pos in positions:
                    context = self.extract_context(title + " " + content, pos)
                    match_score = category.weight * category.weight
                    
                    # Position boost (earlier mentions get higher score)
                    position_boost = 1.0 - (pos / len(full_text)) * 0.3
                    final_score = match_score * position_boost
                    
                    keyword_match = KeywordMatch(
                        keyword=keyword,
                        category=category_name,
                        score=final_score,
                        position=pos,
                        context=context,
                        weight=category.weight
                    )
                    matches.append(keyword_match)
                    total_category_score += final_score
            
            if matches:
                category_matches[category_name] = matches
                category_scores[category_name] = min(total_category_score, 1.0)
        
        # Calculate scores
        total_score = sum(category_scores.values())
        intersection_score = self.calculate_intersection_score(category_matches)
        region_boost = self.calculate_region_boost(title + " " + content, region)
        
        # Final score combines all factors
        final_score = (total_score * 0.6 + intersection_score * 0.3) * region_boost
        final_score = min(final_score, 1.0)
        
        # Flatten all matches for result
        all_matches = []
        for matches in category_matches.values():
            all_matches.extend(matches)
        
        # Sort matches by score
        all_matches.sort(key=lambda x: x.score, reverse=True)
        
        execution_time = time.time() - start_time
        
        return MultiKeywordResult(
            is_relevant=final_score >= min_score,
            total_score=total_score,
            matches=all_matches,
            category_scores=category_scores,
            intersection_score=intersection_score,
            region_boost=region_boost,
            final_score=final_score,
            execution_time=execution_time
        )
    
    def filter_articles(
        self, 
        articles: List[Dict[str, Any]], 
        filter_criteria: Dict[str, Any],
        min_score: float = 0.1
    ) -> List[Tuple[Dict[str, Any], MultiKeywordResult]]:
        """
        Filter articles based on multi-criteria with enhanced scoring.
        
        Args:
            articles: List of articles with title, content, region
            filter_criteria: Dict with categories and required combinations
            min_score: Minimum score threshold
            
        Returns:
            List of (article, analysis_result) tuples for matching articles
        """
        filtered_results = []
        
        # Build categories from filter criteria
        categories = {}
        for category_name, keywords in filter_criteria.get('categories', {}).items():
            categories[category_name] = KeywordCategory(
                name=category_name,
                keywords=keywords,
                weight=filter_criteria.get('weights', {}).get(category_name, 1.0)
            )
        
        for article in articles:
            result = self.analyze_multi_keywords(
                title=article.get('title', ''),
                content=article.get('content', ''),
                categories=categories,
                region=article.get('region', 'global'),
                min_score=min_score
            )
            
            # Check if article meets criteria
            if result.is_relevant:
                # Additional filtering based on required combinations
                required_combinations = filter_criteria.get('required_combinations', [])
                if required_combinations:
                    matched_categories = set(result.category_scores.keys())
                    for combo in required_combinations:
                        if isinstance(combo, list):
                            if set(combo).issubset(matched_categories):
                                filtered_results.append((article, result))
                                break
                        elif combo in matched_categories:
                            filtered_results.append((article, result))
                            break
                else:
                    filtered_results.append((article, result))
        
        # Sort by final score
        filtered_results.sort(key=lambda x: x[1].final_score, reverse=True)
        
        return filtered_results
    
    def create_ai_insurance_uk_filter(self) -> Dict[str, Any]:
        """Create optimized filter for AI + insurance in UK."""
        return {
            'categories': {
                'ai': self.categories['ai'].keywords,
                'insurance': self.categories['insurance'].keywords
            },
            'weights': {
                'ai': 1.0,
                'insurance': 1.2  # Boost insurance keywords for this use case
            },
            'required_combinations': [['ai', 'insurance']],
            'region': 'uk'
        }
    
    def create_ai_healthcare_us_filter(self) -> Dict[str, Any]:
        """Create optimized filter for AI + healthcare in US."""
        return {
            'categories': {
                'ai': self.categories['ai'].keywords,
                'healthcare': self.categories['healthcare'].keywords
            },
            'weights': {
                'ai': 1.0,
                'healthcare': 1.1  # Boost healthcare keywords
            },
            'required_combinations': [['ai', 'healthcare']],
            'region': 'us'
        }
    
    def create_ml_fintech_eu_filter(self) -> Dict[str, Any]:
        """Create optimized filter for ML + fintech in EU."""
        ml_keywords = ['ML', 'machine learning', 'deep learning', 'neural network', 'algorithm']
        return {
            'categories': {
                'ml': ml_keywords,
                'fintech': self.categories['fintech'].keywords
            },
            'weights': {
                'ml': 1.0,
                'fintech': 1.1
            },
            'required_combinations': [['ml', 'fintech']],
            'region': 'eu'
        }
    
    def get_coverage_report(self, results: List[Tuple[Dict[str, Any], MultiKeywordResult]]) -> Dict[str, Any]:
        """Generate comprehensive coverage report from filtered results."""
        if not results:
            return {"total_articles": 0, "categories": {}, "intersections": {}}
        
        # Category statistics
        category_stats = defaultdict(lambda: {"count": 0, "total_score": 0, "keywords": Counter()})
        intersection_stats = defaultdict(int)
        region_stats = defaultdict(int)
        score_distribution = [0, 0, 0, 0]  # 0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0
        
        for article, result in results:
            # Category statistics
            for category, score in result.category_scores.items():
                category_stats[category]["count"] += 1
                category_stats[category]["total_score"] += score
                
                # Count keywords
                category_keywords = [m.keyword for m in result.matches if m.category == category]
                category_stats[category]["keywords"].update(category_keywords)
            
            # Intersection statistics
            categories = set(result.category_scores.keys())
            if len(categories) >= 2:
                intersection_key = " + ".join(sorted(categories))
                intersection_stats[intersection_key] += 1
            
            # Region statistics
            region = article.get('region', 'global')
            region_stats[region] += 1
            
            # Score distribution
            score = result.final_score
            if score <= 0.25:
                score_distribution[0] += 1
            elif score <= 0.5:
                score_distribution[1] += 1
            elif score <= 0.75:
                score_distribution[2] += 1
            else:
                score_distribution[3] += 1
        
        # Calculate averages
        for category_stats_item in category_stats.values():
            if category_stats_item["count"] > 0:
                category_stats_item["avg_score"] = category_stats_item["total_score"] / category_stats_item["count"]
        
        return {
            "total_articles": len(results),
            "categories": dict(category_stats),
            "intersections": dict(intersection_stats),
            "regions": dict(region_stats),
            "score_distribution": score_distribution,
            "avg_score": sum(r.final_score for _, r in results) / len(results) if results else 0
        }


# Example usage and testing functions
def test_enhanced_collector():
    """Test the enhanced multi-keyword collector."""
    collector = EnhancedMultiKeywordCollector(performance_mode=True)
    
    # Test article
    test_article = {
        "title": "AI Revolution in UK Insurance Underwriting",
        "content": "London-based insurtech companies are deploying artificial intelligence and machine learning algorithms for automated underwriting and risk assessment in the Lloyd's market. British firms are leading the way in claims processing automation.",
        "region": "uk"
    }
    
    # Test AI + insurance analysis
    result = collector.analyze_multi_keywords(
        title=test_article["title"],
        content=test_article["content"],
        region=test_article["region"]
    )
    
    print("Enhanced Multi-Keyword Analysis Results:")
    print(f"Is Relevant: {result.is_relevant}")
    print(f"Final Score: {result.final_score:.3f}")
    print(f"Total Score: {result.total_score:.3f}")
    print(f"Intersection Score: {result.intersection_score:.3f}")
    print(f"Region Boost: {result.region_boost:.3f}")
    print(f"Execution Time: {result.execution_time:.4f}s")
    print(f"Category Scores: {result.category_scores}")
    
    print(f"\nTop Matches:")
    for match in result.matches[:5]:
        print(f"  {match.keyword} ({match.category}): {match.score:.3f}")
        print(f"    Context: {match.context}")
    
    # Test filtering
    articles = [test_article]
    filter_criteria = collector.create_ai_insurance_uk_filter()
    filtered = collector.filter_articles(articles, filter_criteria)
    
    print(f"\nFiltering Results:")
    print(f"Found {len(filtered)} matching articles")
    
    # Test coverage report
    report = collector.get_coverage_report(filtered)
    print(f"\nCoverage Report:")
    print(f"Total articles: {report['total_articles']}")
    print(f"Average score: {report['avg_score']:.3f}")
    print(f"Categories: {list(report['categories'].keys())}")
    print(f"Intersections: {list(report['intersections'].keys())}")


if __name__ == "__main__":
    test_enhanced_collector()
