#!/usr/bin/env python3
"""
Comprehensive search utility with multiple flexible search strategies.
"""

import sys
import re
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from src.ai_news.config import Config
from src.ai_news.database import Database

class FlexibleSearchEngine:
    """Advanced search engine with multiple strategies."""
    
    def __init__(self, database):
        self.db = database
        self.ai_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'deep learning',
            'neural network', 'llm', 'chatgpt', 'openai', 'anthropic', 'claude',
            'automation', 'algorithm', 'predictive', 'analytics'
        ]
    
    def search(self, query, strategy='smart', **filters):
        """Main search method with strategy selection."""
        
        # Get base articles
        articles = self.db.get_articles(
            limit=filters.get('limit', 1000),
            ai_only=filters.get('ai_only', False),
            region=filters.get('region')
        )
        
        # Apply search strategy
        if strategy == 'smart':
            results = self.smart_search(articles, query, **filters)
        elif strategy == 'boolean':
            results = self.boolean_search(articles, query, **filters)
        elif strategy == 'fuzzy':
            results = self.fuzzy_search(articles, query, **filters)
        elif strategy == 'semantic':
            results = self.semantic_search(articles, query, **filters)
        elif strategy == 'pattern':
            results = self.pattern_search(articles, query, **filters)
        else:
            results = self.basic_search(articles, query, **filters)
        
        return results[:filters.get('limit', 20)]
    
    def smart_search(self, articles, query, **filters):
        """Smart search that combines multiple techniques."""
        results = []
        
        for article in articles:
            score = 0
            details = []
            
            # Text preparation
            title = article.title.lower()
            content = article.content.lower()
            summary = article.summary.lower()
            full_text = f"{title} {content} {summary}"
            
            # 1. Exact phrase matches (highest weight)
            if query.lower() in full_text:
                score += 50
                details.append("Exact phrase match")
            
            # 2. Enhanced individual keyword matches
            query_words = query.lower().split()
            matched_words = []
            for word in query_words:
                if self._enhanced_keyword_match(word, full_text):
                    matched_words.append(word)
            if matched_words:
                score += len(matched_words) * 10
                details.append(f"Keywords: {', '.join(matched_words)}")
            
            # 3. Title matches (bonus)
            title_matches = [word for word in query_words if word in title]
            if title_matches:
                score += len(title_matches) * 15
                details.append(f"Title matches: {', '.join(title_matches)}")
            
            # 4. AI relevance bonus
            if article.ai_relevant:
                score += 20
                details.append("AI-relevant")
            
            # 5. Source authority bonus
            if self.is_authoritative_source(article.source_name):
                score += 10
                details.append("Authoritative source")
            
            # 6. Recency bonus
            if article.published_at:
                days_old = (datetime.now().replace(tzinfo=None) - 
                           article.published_at.replace(tzinfo=None)).days
                if days_old <= 7:
                    score += 15
                    details.append("Recent article")
                elif days_old <= 30:
                    score += 5
                    details.append("Recent month")
            
            if score > 0:
                article.search_score = score
                article.search_details = details
                results.append(article)
        
        # Sort by score
        results.sort(key=lambda x: x.search_score, reverse=True)
        return results
    
    def boolean_search(self, articles, query, **filters):
        """Boolean search with AND, OR, NOT operators."""
        results = []
        
        # Parse boolean expression (simplified)
        and_terms = []
        or_terms = []
        not_terms = []
        
        # Simple parsing - in real implementation, use proper boolean parser
        if ' AND ' in query.upper():
            and_terms = [term.strip() for term in query.upper().split(' AND ')]
        elif ' OR ' in query.upper():
            or_terms = [term.strip() for term in query.upper().split(' OR ')]
        else:
            and_terms = [query.upper()]
        
        # Extract NOT terms
        for term_list in [and_terms, or_terms]:
            new_terms = []
            for term in term_list:
                if ' NOT ' in term:
                    parts = term.split(' NOT ')
                    if parts[0]:
                        new_terms.append(parts[0])
                    not_terms.extend(parts[1:])
                else:
                    new_terms.append(term)
            term_list[:] = new_terms
        
        for article in articles:
            text = f"{article.title} {article.content} {article.summary}".lower()
            
            # Check AND terms (all must match)
            and_match = all(term.lower() in text for term in and_terms) if and_terms else True
            
            # Check OR terms (any must match)
            or_match = any(term.lower() in text for term in or_terms) if or_terms else True
            
            # Check NOT terms (none must match)
            not_match = not any(term.lower() in text for term in not_terms) if not_terms else True
            
            if and_match and or_match and not_match:
                article.search_score = 50
                article.search_details = [f"Boolean: {' AND '.join(and_terms)}"]
                results.append(article)
        
        return results
    
    def fuzzy_search(self, articles, query, **filters):
        """Fuzzy search with typo tolerance and variations."""
        results = []
        
        # Generate variations of the query
        variations = self.generate_variations(query.lower())
        
        for article in articles:
            text = f"{article.title} {article.content} {article.summary}".lower()
            
            best_match = 0
            best_variation = ""
            
            for variation in variations:
                if variation in text:
                    # Calculate match quality
                    match_quality = len(variation) / len(query)
                    if match_quality > best_match:
                        best_match = match_quality
                        best_variation = variation
            
            if best_match > 0.5:  # Threshold for fuzzy match
                article.search_score = int(best_match * 100)
                article.search_details = [f"Fuzzy: {best_variation} ({best_match:.2f})"]
                results.append(article)
        
        results.sort(key=lambda x: x.search_score, reverse=True)
        return results
    
    def semantic_search(self, articles, query, **filters):
        """Semantic search using keyword categories and concepts."""
        results = []
        
        # Define semantic categories
        categories = {
            'technology': ['tech', 'software', 'digital', 'platform', 'app', 'system'],
            'business': ['business', 'company', 'corporation', 'startup', 'enterprise', 'firm'],
            'finance': ['finance', 'financial', 'investment', 'funding', 'capital', 'money'],
            'healthcare': ['health', 'medical', 'healthcare', 'pharma', 'hospital', 'clinic'],
            'ai_ml': self.ai_keywords,
            'regulation': ['regulation', 'compliance', 'policy', 'law', 'legal', 'government'],
            'innovation': ['innovation', 'breakthrough', 'research', 'development', 'patent']
        }
        
        # Find matching categories
        query_lower = query.lower()
        matching_categories = []
        
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                matching_categories.append(category)
        
        for article in articles:
            text = f"{article.title} {article.content} {article.summary}".lower()
            
            score = 0
            matched_categories = []
            
            # Check each matching category
            for category in matching_categories:
                category_keywords = categories[category]
                matches = sum(1 for kw in category_keywords if kw in text)
                if matches > 0:
                    score += matches * 10
                    matched_categories.append(category)
            
            if score > 0:
                article.search_score = score
                article.search_details = [f"Semantic: {', '.join(matched_categories)}"]
                results.append(article)
        
        results.sort(key=lambda x: x.search_score, reverse=True)
        return results
    
    def pattern_search(self, articles, query, **filters):
        """Pattern-based search using regex."""
        results = []
        
        try:
            # Convert query to regex pattern
            pattern = re.compile(query, re.IGNORECASE)
            
            for article in articles:
                text = f"{article.title} {article.content} {article.summary}"
                
                matches = pattern.findall(text)
                if matches:
                    article.search_score = len(matches) * 20
                    article.search_details = [f"Pattern: {len(matches)} matches"]
                    results.append(article)
        
        except re.error:
            # Fallback to basic search if regex is invalid
            return self.basic_search(articles, query, **filters)
        
        results.sort(key=lambda x: x.search_score, reverse=True)
        return results
    
    def basic_search(self, articles, query, **filters):
        """Basic substring search."""
        results = []
        query_lower = query.lower()
        
        for article in articles:
            text = f"{article.title} {article.content} {article.summary}".lower()
            
            if query_lower in text:
                article.search_score = 30
                article.search_details = ["Basic match"]
                results.append(article)
        
        return results
    
    def generate_variations(self, query):
        """Generate query variations for fuzzy matching."""
        variations = [query]
        
        # Common typos and variations
        typo_map = {
            'ai': ['a.i.', 'artificial intelligence'],
            'insurance': ['insurace', 'insuranc', 'insurence'],
            'technology': ['tech', 'technlogy', 'tecnology'],
            'company': ['companies', 'corp', 'corporation'],
            'fintech': ['fin tech', 'financial technology'],
            'healthcare': ['health care', 'medical']
        }
        
        for word, variations_list in typo_map.items():
            if word in query:
                for variation in variations_list:
                    variations.append(query.replace(word, variation))
        
        return variations
    
    def is_authoritative_source(self, source_name):
        """Check if source is considered authoritative."""
        authoritative = [
            'bbc', 'reuters', 'associated press', 'bloomberg', 'wall street journal',
            'financial times', 'techcrunch', 'venturebeat', 'nature', 'science',
            'mit', 'stanford', 'harvard'
        ]
        return any(auth in source_name.lower() for auth in authoritative)
    
    def _enhanced_keyword_match(self, keyword: str, text: str) -> bool:
        """Enhanced keyword matching with word boundaries and fuzzy matching."""
        keyword_lower = keyword.lower()
        text_lower = text.lower()
        
        # 1. Direct word boundary matching
        if self._matches_word_boundary(keyword_lower, text_lower):
            return True
            
        # 2. Check common AI variations
        keyword_variations = {
            'ai': ['ai', 'a.i.', 'artificial intelligence'],
            'ml': ['ml', 'machine learning'],
            'dl': ['dl', 'deep learning'],
            'nlp': ['nlp', 'natural language processing'],
            'llm': ['llm', 'large language model'],
            'gpt': ['gpt', 'chatgpt', 'gpt-3', 'gpt-4'],
        }
        
        if keyword_lower in keyword_variations:
            for variation in keyword_variations[keyword_lower]:
                if self._matches_word_boundary(variation, text_lower):
                    return True
                
        # 3. Basic fuzzy matching for typos (threshold 0.85)
        if self._fuzzy_match(keyword_lower, text_lower, 0.85):
            return True
            
        return False
    
    def _matches_word_boundary(self, keyword: str, text: str) -> bool:
        """Check if keyword appears as a whole word in text."""
        # For patterns with dots, use more flexible matching
        if '.' in keyword:
            # Handle A.I., L.L.M., etc.
            escaped_keyword = re.escape(keyword)
            pattern = rf'(?<!\w){escaped_keyword}(?!\w)'
        else:
            # Standard word boundary matching
            escaped_keyword = re.escape(keyword)
            pattern = rf'\b{escaped_keyword}\b'
        return bool(re.search(pattern, text, re.IGNORECASE))
    
    def _fuzzy_match(self, keyword: str, text: str, threshold: float = 0.85) -> bool:
        """Basic fuzzy matching using SequenceMatcher."""
        words = text.split()
        for word in words:
            similarity = SequenceMatcher(None, keyword, word).ratio()
            if similarity >= threshold:
                return True
        return False

def format_search_results(results, show_details=False, max_snippet_length=100):
    """Format search results for display."""
    if not results:
        print("🔍 No articles found matching your search criteria.")
        return
    
    print(f"\n🔍 Found {len(results)} articles:")
    print("=" * 80)
    
    for i, article in enumerate(results, 1):
        # Article header
        relevance_indicator = "🤖" if article.ai_relevant else "  "
        score_text = f"[Score: {article.search_score:02d}]" if hasattr(article, 'search_score') else ""
        
        print(f"{i}. {relevance_indicator} {article.title} {score_text}")
        
        # Article metadata
        date_str = article.published_at.strftime("%Y-%m-%d") if article.published_at else "Unknown"
        print(f"   📅 {date_str} | 📰 {article.source_name} | 🏷️  {article.category}")
        
        # Search details
        if show_details and hasattr(article, 'search_details'):
            print(f"   🔍 {' | '.join(article.search_details)}")
        
        # Content snippet
        snippet = article.summary or article.content[:max_snippet_length]
        if len(snippet) > max_snippet_length:
            snippet = snippet[:max_snippet_length] + "..."
        print(f"   📄 {snippet}")
        
        print(f"   🔗 {article.url}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Comprehensive flexible search for AI News')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--strategy', choices=['smart', 'boolean', 'fuzzy', 'semantic', 'pattern', 'basic'], 
                       default='smart', help='Search strategy')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--region', choices=['us', 'uk', 'eu', 'apac', 'global'], help='Filter by region')
    parser.add_argument('--limit', type=int, default=20, help='Number of articles to show')
    parser.add_argument('--ai-only', action='store_true', help='Show only AI-relevant articles')
    parser.add_argument('--show-details', action='store_true', help='Show detailed match information')
    
    args = parser.parse_args()
    
    # Load configuration and database
    config_path = Path(args.config)
    config = Config.load(config_path)
    database = Database(config.database_path)
    
    # Create search engine
    search_engine = FlexibleSearchEngine(database)
    
    # Perform search
    filters = {
        'limit': args.limit,
        'ai_only': args.ai_only,
        'region': args.region
    }
    
    print(f"🔍 Searching with {args.strategy.upper()} strategy: '{args.query}'")
    if args.region:
        print(f"🌍 Region: {args.region.upper()}")
    if args.ai_only:
        print(f"🤖 AI-relevant only")
    
    results = search_engine.search(args.query, args.strategy, **filters)
    
    # Display results
    format_search_results(results, args.show_details)

if __name__ == '__main__':
    main()