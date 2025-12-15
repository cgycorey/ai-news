#!/usr/bin/env python3
"""
Advanced flexible search utility for AI News with multiple search strategies.
"""

import sys
import argparse
import re
from pathlib import Path
from datetime import datetime, timedelta
from src.ai_news.config import Config
from src.ai_news.database import Database

def flexible_search(database, search_strategy, **kwargs):
    """
    Advanced search with multiple strategies.
    """
    articles = database.get_articles(
        limit=kwargs.get('limit', 1000), 
        ai_only=kwargs.get('ai_only', False), 
        region=kwargs.get('region')
    )
    
    results = []
    
    for article in articles:
        search_text = {
            'title': article.title.lower(),
            'content': article.content.lower(),
            'summary': article.summary.lower(),
            'combined': f"{article.title} {article.content} {article.summary}".lower()
        }
        
        # Extract search parameters before passing to strategy function
        search_params = {}
        if search_strategy == 'regex':
            search_params['pattern'] = kwargs.get('pattern', '')
        elif search_strategy == 'phrase':
            search_params['phrase'] = kwargs.get('phrase', '')
        elif search_strategy == 'proximity':
            search_params['keywords'] = kwargs.get('keywords', [])
            search_params['max_distance'] = kwargs.get('max_distance', 50)
        elif search_strategy == 'field_specific':
            search_params['query'] = kwargs.get('query', '')
            search_params['fields'] = kwargs.get('fields', ['title'])
        elif search_strategy == 'date_range':
            search_params['start_date'] = kwargs.get('start_date')
            search_params['end_date'] = kwargs.get('end_date')
        elif search_strategy == 'source_filter':
            search_params['sources'] = kwargs.get('sources', [])
        elif search_strategy == 'category_filter':
            search_params['categories'] = kwargs.get('categories', [])
        elif search_strategy == 'ai_relevance_threshold':
            search_params['min_relevance'] = kwargs.get('min_relevance', 0.5)
        
        match_result = evaluate_search_strategy(search_strategy, search_text, article, **search_params)
        if match_result['matched']:
            article.search_score = match_result['score']
            article.match_details = match_result['details']
            results.append(article)
    
    # Sort by search score
    results.sort(key=lambda x: x.search_score, reverse=True)
    return results[:kwargs.get('limit', 20)]

def evaluate_search_strategy(strategy, search_text, article, **kwargs):
    """Evaluate different search strategies."""
    result = {'matched': False, 'score': 0, 'details': []}
    
    if strategy == 'regex':
        return regex_search(search_text, kwargs.get('pattern', ''), **kwargs)
    elif strategy == 'phrase':
        return phrase_search(search_text, kwargs.get('phrase', ''), **kwargs)
    elif strategy == 'proximity':
        return proximity_search(search_text, kwargs.get('keywords', []), **kwargs)
    elif strategy == 'field_specific':
        return field_specific_search(search_text, kwargs.get('query', ''), kwargs.get('fields', ['title']), **kwargs)
    elif strategy == 'date_range':
        return date_range_search(article, kwargs.get('start_date'), kwargs.get('end_date'), **kwargs)
    elif strategy == 'source_filter':
        return source_filter_search(article, kwargs.get('sources', []), **kwargs)
    elif strategy == 'category_filter':
        return category_filter_search(article, kwargs.get('categories', []), **kwargs)
    elif strategy == 'ai_relevance_threshold':
        return ai_relevance_search(article, kwargs.get('min_relevance', 0.5), **kwargs)
    else:
        return result

def regex_search(search_text, pattern, **kwargs):
    """Regular expression search."""
    result = {'matched': False, 'score': 0, 'details': []}
    
    try:
        regex = re.compile(pattern, re.IGNORECASE)
        
        for field, text in search_text.items():
            matches = regex.findall(text)
            if matches:
                result['matched'] = True
                result['score'] += len(matches) * 10
                result['details'].append(f"{field}: {len(matches)} regex matches")
        
        if result['matched']:
            result['score'] = min(result['score'], 100)  # Cap score
            
    except re.error as e:
        result['details'].append(f"Regex error: {e}")
    
    return result

def phrase_search(search_text, phrase, **kwargs):
    """Exact phrase search."""
    result = {'matched': False, 'score': 0, 'details': []}
    phrase_lower = phrase.lower()
    
    for field, text in search_text.items():
        if phrase_lower in text:
            result['matched'] = True
            # Score based on field importance (title > summary > content)
            field_weights = {'title': 30, 'summary': 20, 'content': 10, 'combined': 5}
            result['score'] += field_weights.get(field, 5)
            result['details'].append(f"Phrase found in {field}")
    
    return result

def proximity_search(search_text, keywords, **kwargs):
    """Proximity search - keywords must be within N words of each other."""
    result = {'matched': False, 'score': 0, 'details': []}
    max_distance = kwargs.get('max_distance', 50)
    
    if len(keywords) < 2:
        return result
    
    text = search_text['combined']
    words = text.split()
    
    # Find positions of each keyword
    keyword_positions = {}
    for i, word in enumerate(words):
        for keyword in keywords:
            if keyword.lower() in word.lower():
                if keyword not in keyword_positions:
                    keyword_positions[keyword] = []
                keyword_positions[keyword].append(i)
    
    # Check if all keywords are within max_distance of each other
    if len(keyword_positions) == len(keywords):
        all_positions = []
        for positions in keyword_positions.values():
            all_positions.extend(positions)
        
        if len(all_positions) >= 2:
            min_pos = min(all_positions)
            max_pos = max(all_positions)
            if max_pos - min_pos <= max_distance:
                result['matched'] = True
                result['score'] = max(0, 50 - (max_pos - min_pos))
                result['details'].append(f"Keywords within {max_pos - min_pos} words")
    
    return result

def field_specific_search(search_text, query, fields, **kwargs):
    """Search in specific fields only."""
    result = {'matched': False, 'score': 0, 'details': []}
    query_lower = query.lower()
    
    for field in fields:
        if field in search_text:
            text = search_text[field]
            if query_lower in text:
                result['matched'] = True
                field_weights = {'title': 30, 'summary': 20, 'content': 10}
                result['score'] += field_weights.get(field, 5)
                result['details'].append(f"Found in {field}")
    
    return result

def date_range_search(article, start_date, end_date, **kwargs):
    """Filter by publication date range."""
    result = {'matched': False, 'score': 0, 'details': []}
    
    if not article.published_at:
        return result
    
    article_date = article.published_at.date()
    
    if start_date and end_date:
        if start_date <= article_date <= end_date:
            result['matched'] = True
            result['score'] = 50
            result['details'].append(f"Date {article_date} in range")
    elif start_date and article_date >= start_date:
        result['matched'] = True
        result['score'] = 50
        result['details'].append(f"Date {article_date} >= {start_date}")
    elif end_date and article_date <= end_date:
        result['matched'] = True
        result['score'] = 50
        result['details'].append(f"Date {article_date} <= {end_date}")
    
    return result

def source_filter_search(article, sources, **kwargs):
    """Filter by specific sources."""
    result = {'matched': False, 'score': 0, 'details': []}
    
    for source in sources:
        if source.lower() in article.source_name.lower():
            result['matched'] = True
            result['score'] = 30
            result['details'].append(f"Source: {article.source_name}")
            break
    
    return result

def category_filter_search(article, categories, **kwargs):
    """Filter by article categories."""
    result = {'matched': False, 'score': 0, 'details': []}
    
    for category in categories:
        if category.lower() in article.category.lower():
            result['matched'] = True
            result['score'] = 20
            result['details'].append(f"Category: {article.category}")
            break
    
    return result

def ai_relevance_search(article, min_relevance, **kwargs):
    """Filter by AI relevance score (simulated)."""
    result = {'matched': False, 'score': 0, 'details': []}
    
    # Simulate AI relevance score based on keywords
    ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'llm', 'chatgpt', 'openai', 'anthropic']
    
    text = f"{article.title} {article.content} {article.summary}".lower()
    keyword_count = sum(1 for keyword in ai_keywords if keyword in text)
    simulated_score = min(keyword_count * 10, 100)
    
    if simulated_score >= min_relevance * 100:
        result['matched'] = True
        result['score'] = simulated_score
        result['details'].append(f"AI relevance: {simulated_score}%")
    
    return result

def format_results(articles, show_details=False):
    """Format search results for display."""
    if not articles:
        print("No articles found matching your criteria.")
        return
    
    print(f"\n🔍 Found {len(articles)} articles:")
    print("=" * 80)
    
    for i, article in enumerate(articles, 1):
        relevance_indicator = "🤖" if article.ai_relevant else "  "
        score_text = f"[Score: {article.search_score:02d}]" if hasattr(article, 'search_score') else ""
        
        print(f"{i}. {relevance_indicator} {article.title} {score_text}")
        print(f"   Source: {article.source_name} | {article.published_at or 'Unknown date'}")
        print(f"   Category: {article.category}")
        
        if show_details and hasattr(article, 'match_details'):
            print(f"   Match details: {' | '.join(article.match_details)}")
        
        print(f"   URL: {article.url}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Advanced flexible search with multiple strategies')
    
    # Strategy selection
    strategy_group = parser.add_mutually_exclusive_group(required=True)
    strategy_group.add_argument('--regex', help='Regular expression search')
    strategy_group.add_argument('--phrase', help='Exact phrase search')
    strategy_group.add_argument('--proximity', nargs='+', help='Proximity search (keywords within N words)')
    strategy_group.add_argument('--fields', help='Field-specific search')
    strategy_group.add_argument('--date-range', action='store_true', help='Filter by date range')
    strategy_group.add_argument('--sources', nargs='+', help='Filter by sources')
    strategy_group.add_argument('--categories', nargs='+', help='Filter by categories')
    strategy_group.add_argument('--ai-threshold', type=float, help='Filter by AI relevance threshold (0-1)')
    
    # Common options
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--region', choices=['us', 'uk', 'eu', 'apac', 'global'], help='Filter by region')
    parser.add_argument('--limit', type=int, default=20, help='Number of articles to show')
    parser.add_argument('--ai-only', action='store_true', help='Show only AI-relevant articles')
    parser.add_argument('--show-details', action='store_true', help='Show detailed match information')
    
    # Strategy-specific options
    parser.add_argument('--max-distance', type=int, default=50, help='Max word distance for proximity search')
    parser.add_argument('--search-fields', nargs='+', default=['title'], help='Fields to search in (for --fields)')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD) for date range')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD) for date range')
    
    args = parser.parse_args()
    
    # Load configuration and database
    config_path = Path(args.config)
    config = Config.load(config_path)
    database = Database(config.database_path)
    
    # Determine search strategy and parameters
    search_kwargs = {
        'limit': args.limit,
        'ai_only': args.ai_only,
        'region': args.region
    }
    
    if args.regex:
        search_strategy = 'regex'
        search_kwargs['pattern'] = args.regex
        print(f"🔍 Regex Search: {args.regex}")
        
    elif args.phrase:
        search_strategy = 'phrase'
        search_kwargs['phrase'] = args.phrase
        print(f"🔍 Phrase Search: \"{args.phrase}\"")
        
    elif args.proximity:
        search_strategy = 'proximity'
        search_kwargs['keywords'] = args.proximity
        search_kwargs['max_distance'] = args.max_distance
        print(f"🔍 Proximity Search: {args.proximity} (within {args.max_distance} words)")
        
    elif args.fields:
        search_strategy = 'field_specific'
        search_kwargs['query'] = args.fields[0]  # First item is query
        search_kwargs['fields'] = args.search_fields
        print(f"🔍 Field Search: \"{args.fields[0]}\" in {args.search_fields}")
        
    elif args.date_range:
        search_strategy = 'date_range'
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date() if args.start_date else None
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date() if args.end_date else None
        search_kwargs['start_date'] = start_date
        search_kwargs['end_date'] = end_date
        print(f"🔍 Date Range: {args.start_date} to {args.end_date}")
        
    elif args.sources:
        search_strategy = 'source_filter'
        search_kwargs['sources'] = args.sources
        print(f"🔍 Source Filter: {args.sources}")
        
    elif args.categories:
        search_strategy = 'category_filter'
        search_kwargs['categories'] = args.categories
        print(f"🔍 Category Filter: {args.categories}")
        
    elif args.ai_threshold:
        search_strategy = 'ai_relevance_threshold'
        search_kwargs['min_relevance'] = args.ai_threshold
        print(f"🔍 AI Relevance Threshold: {args.ai_threshold}")
        
    # Search and display results
    articles = flexible_search(database, search_strategy, **search_kwargs)
    format_results(articles, args.show_details)

if __name__ == '__main__':
    main()