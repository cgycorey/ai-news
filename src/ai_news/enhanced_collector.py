"""Enhanced multi-keyword collector for AI News system."""

import re
import time
from typing import List, Dict, Tuple, Any, Set, Optional, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import logging

from .config import FeedConfig
from .database import Article
from .collector import SimpleCollector

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
                    'insurance', 'insurtech', 'insur tech', 'insurance technology', 'underwriting', 'claims', 'risk', 
                    'premium', 'coverage', 'deductible', 'policy', 'actuarial', 'actuarial science',
                    'reinsurance', 'broker', 'Lloyd\'s', 'insurance startup', 'digital insurance',
                    'claims processing', 'risk assessment', 'policy management'
                ],
                weight=0.8
            ),
            'healthcare': KeywordCategory(
                name='healthcare',
                keywords=[
                    'healthcare', 'health care', 'medical', 'diagnostics', 'medicine', 'clinical',
                    'hospital', 'biotech', 'biotechnology', 'pharmaceutical', 'pharma', 'drug discovery',
                    'medical imaging', 'telemedicine', 'health', 'health tech', 'health technology',
                    'digital health', 'medical technology', 'medtech', 'life sciences', 'healthcare IT',
                    'clinical trials', 'medical devices', 'personalized medicine'
                ],
                weight=0.8
            ),
            'fintech': KeywordCategory(
                name='fintech',
                keywords=[
                    'fintech', 'fin tech', 'financial technology', 'banking', 'financial', 'trading', 'payments',
                    'fraud detection', 'regtech', 'regulatory technology', 'compliance', 'anti-money laundering',
                    'AML', 'digital banking', 'neobank', 'neo bank', 'challenger bank', 'online banking',
                    'cryptocurrency', 'crypto', 'blockchain', 'wealthtech', 'wealth management',
                    'paytech', 'payment technology', 'insurtech', 'lending', 'credit scoring'
                ],
                weight=0.8
            ),
            'ml': KeywordCategory(
                name='ml',
                keywords=[
                    'ML', 'machine learning', 'deep learning', 'neural network', 
                    'algorithm', 'algorithms', 'predictive analytics', 'data science',
                    'artificial intelligence', 'AI', 'model training', 'supervised learning'
                ],
                weight=1.0,
                variations=self.default_variations
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
    
    def build_keyword_index(self, categories: Dict[str, Union[KeywordCategory, List[str]]]):
        """Build performance index for keyword matching."""
        self.keyword_index = {}
        
        for category_name, category in categories.items():
            keywords, weight, variations = self._extract_category_data(category)
            
            for keyword in keywords:
                # Index the base keyword
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append((category_name, weight))
                
                # Index variations
                if keyword.lower() in variations:
                    for variation in variations[keyword.lower()]:
                        if variation not in self.keyword_index:
                            self.keyword_index[variation] = []
                        self.keyword_index[variation].append((category_name, weight * 0.9))
    
    def _extract_category_data(self, category: Union[KeywordCategory, List[str]]) -> Tuple[List[str], float, Dict[str, List[str]]]:
        """Safely extract keyword data from both KeywordCategory objects and plain lists.
        
        Args:
            category: Either a KeywordCategory object or a list of keywords
            
        Returns:
            Tuple of (keywords, weight, variations)
        """
        if hasattr(category, 'keywords'):
            # KeywordCategory object
            keywords = category.keywords
            weight = getattr(category, 'weight', 1.0)
            variations = getattr(category, 'variations', {})
        else:
            # Plain list or single keyword
            keywords = category if isinstance(category, list) else [category]
            weight = 1.0
            variations = {}
        
        return keywords, weight, variations
    
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
        base_score = min(len(categories_with_matches) * 0.25, 1.0)
        
        # Bonus for specific high-value combinations
        high_value_combinations = [
            {'ai', 'insurance'},
            {'ai', 'healthcare'},
            {'ai', 'fintech'},
            {'ai', 'ml'},  # AI + ML is valuable
            {'insurance', 'healthcare'},
            {'fintech', 'insurance'},
            {'healthcare', 'biotech'},  # Healthcare + biotech combo
            {'fintech', 'regtech'}  # Finance + regulatory combo
        ]
        
        category_set = set(categories_with_matches)
        bonus_score = 0.0
        for combo in high_value_combinations:
            if combo.issubset(category_set):
                bonus_score += 0.15  # Add bonus for each high-value combo
        
        # Quality bonus: more matches per category indicates stronger relevance
        total_matches = sum(len(matches) for matches in category_matches.values())
        quality_bonus = min(total_matches * 0.02, 0.2)  # Up to 0.2 bonus for many matches
        
        final_score = base_score + bonus_score + quality_bonus
        return min(final_score, 1.0)
    
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
        categories: Dict[str, Union[KeywordCategory, List[str]]] = None,
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
        
        # Combine title, summary, and content for comprehensive analysis
        summary = content[:200] + "..." if len(content) > 200 else content  # Create summary if not provided
        full_text = (title + " " + summary + " " + content).lower()
        
        # Analyze each category
        category_matches = {}
        category_scores = {}
        
        for category_name, category in categories.items():
            matches = []
            total_category_score = 0.0
            
            keywords, weight, variations = self._extract_category_data(category)
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # Direct word boundary matching
                positions = self.matches_word_boundary(keyword_lower, full_text)
                
                # Check variations if no direct matches
                if not positions and keyword_lower in variations:
                    for variation in variations[keyword_lower]:
                        positions.extend(self.matches_word_boundary(variation, full_text))
                
                # Fuzzy matching as fallback
                if not positions:
                    fuzzy_positions = self.fuzzy_match(keyword_lower, full_text, 0.85)
                    positions.extend(fuzzy_positions)
                
                # Create keyword matches
                for pos, end_pos in positions:
                    context = self.extract_context(title + " " + content, pos)
                    match_score = weight * weight
                    
                    # Position boost (earlier mentions get higher score)
                    position_boost = 1.0 - (pos / len(full_text)) * 0.3
                    final_score = match_score * position_boost
                    
                    keyword_match = KeywordMatch(
                        keyword=keyword,
                        category=category_name,
                        score=final_score,
                        position=pos,
                        context=context,
                        weight=weight
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


class EnhancedCollector(SimpleCollector):
    """Enhanced collector with multi-keyword capabilities."""
    
    def __init__(self, database, enable_enhanced: bool = True):
        super().__init__(database)
        self.enable_enhanced = enable_enhanced
        if enable_enhanced:
            self.multi_keyword_collector = EnhancedMultiKeywordCollector(performance_mode=True)
    
    def enhanced_is_ai_relevant(
        self, 
        title: str, 
        content: str, 
        keywords: List[str] = None,
        multi_keyword_criteria: Dict[str, Any] = None,
        region: str = "global"
    ) -> tuple[bool, List[str], Optional[MultiKeywordResult]]:
        """Enhanced AI relevance detection with multi-keyword support."""
        
        # Fall back to basic method if enhanced features are disabled
        if not self.enable_enhanced or not multi_keyword_criteria:
            is_relevant, found_keywords = self.is_ai_relevant(title, content, keywords or [])
            return is_relevant, found_keywords, None
        
        # Use enhanced multi-keyword analysis
        categories = {}
        for category_name, category_keywords in multi_keyword_criteria.get('categories', {}).items():
            categories[category_name] = KeywordCategory(
                name=category_name,
                keywords=category_keywords,
                weight=multi_keyword_criteria.get('weights', {}).get(category_name, 1.0)
            )
        
        result = self.multi_keyword_collector.analyze_multi_keywords(
            title=title,
            content=content,
            categories=categories,
            region=region,
            min_score=multi_keyword_criteria.get('min_score', 0.1)
        )
        
        # Extract found keywords
        found_keywords = [match.keyword for match in result.matches]
        
        # Check required combinations
        if multi_keyword_criteria.get('required_combinations'):
            required_combinations = multi_keyword_criteria['required_combinations']
            matched_categories = set(result.category_scores.keys())
            
            # Check if ALL required combinations are present
            all_combinations_met = True
            for combo in required_combinations:
                if isinstance(combo, list):
                    if not set(combo).issubset(matched_categories):
                        all_combinations_met = False
                        break
                elif combo not in matched_categories:
                    all_combinations_met = False
                    break
            
            # Update result based on required combinations
            result.is_relevant = result.is_relevant and all_combinations_met
        
        return result.is_relevant, found_keywords, result
    
    def fetch_feed_enhanced(
        self, 
        feed_config: FeedConfig, 
        max_articles: int = 50,
        multi_keyword_criteria: Dict[str, Any] = None,
        region: str = "global"
    ) -> List[Article]:
        """Fetch articles with enhanced multi-keyword analysis."""
        articles = []
        
        print(f"  Fetching from {feed_config.name} (enhanced mode)...")
        
        root = self.fetch_rss_feed(feed_config.url)
        if root is None:
            return articles
        
        # Find items (handle both RSS and Atom formats)
        items = []
        channel = root.find('channel')
        if channel is not None:
            items = channel.findall('item')
        else:
            # Atom format
            items = root.findall('entry')
        
        for item in items[:max_articles]:
            try:
                if item.tag == 'entry':  # Atom format
                    data = self.parse_atom_entry(item)
                else:  # RSS format
                    data = self.parse_rss_item(item)
                
                title = data.get('title', '')
                url = data.get('link', '')
                
                if not title or not url:
                    continue
                
                # Clean HTML
                clean_content = self.clean_html(data.get('content', ''))
                
                # Get published date
                published_at = self.parse_date(data.get('date', ''))
                
                # Enhanced AI relevance check
                if multi_keyword_criteria:
                    is_ai, keywords_found, enhanced_result = self.enhanced_is_ai_relevant(
                        title, clean_content, feed_config.ai_keywords, multi_keyword_criteria, region
                    )
                else:
                    is_ai, keywords_found = self.is_ai_relevant(title, clean_content, feed_config.ai_keywords)
                    enhanced_result = None
                
                # Create summary
                summary = self.create_summary(clean_content)
                
                article = Article(
                    title=title,
                    content=clean_content,
                    summary=summary,
                    url=url,
                    author=data.get('author', ''),
                    published_at=published_at,
                    source_name=feed_config.name,
                    category=feed_config.category,
                    region=region,
                    ai_relevant=is_ai,
                    ai_keywords_found=keywords_found
                )
                
                # Store enhanced result if available
                if enhanced_result:
                    article.enhanced_result = enhanced_result
                
                articles.append(article)
                
            except Exception as e:
                print(f"    Error processing article: {e}")
                continue
        
        print(f"  Found {len(articles)} articles (enhanced analysis)")
        return articles