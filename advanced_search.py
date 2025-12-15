#!/usr/bin/env python3
"""
Advanced search utility for AI News with keyword combination support.
"""

import sys
import argparse
from pathlib import Path
from src.ai_news.config import Config
from src.ai_news.database import Database

def search_combined_keywords(database, keywords, require_all=False, ai_only=False, region=None, limit=20):
    """
    Search for articles containing multiple keywords.
    
    Args:
        database: Database instance
        keywords: List of keywords to search for
        require_all: If True, article must contain ALL keywords; if False, ANY keyword
        ai_only: Filter to AI-relevant articles only
        region: Filter by region
        limit: Maximum number of results
    """
    articles = database.get_articles(limit=limit * 2, ai_only=ai_only, region=region)
    
    filtered_articles = []
    for article in articles:
        search_text = f"{article.title} {article.content} {article.summary}".lower()
        
        keyword_matches = []
        for keyword in keywords:
            if keyword.lower() in search_text:
                keyword_matches.append(keyword)
        
        if require_all:
            # Article must contain ALL keywords
            if len(keyword_matches) == len(keywords):
                article.matching_keywords = keyword_matches
                filtered_articles.append(article)
        else:
            # Article must contain ANY keyword
            if keyword_matches:
                article.matching_keywords = keyword_matches
                filtered_articles.append(article)
    
    return filtered_articles[:limit]

def main():
    parser = argparse.ArgumentParser(description='Advanced search with keyword combinations')
    parser.add_argument('keywords', nargs='+', help='Keywords to search for')
    parser.add_argument('--require-all', action='store_true', help='Require ALL keywords (default: ANY keyword)')
    parser.add_argument('--ai-only', action='store_true', help='Show only AI-relevant articles')
    parser.add_argument('--region', choices=['us', 'uk', 'eu', 'apac', 'global'], help='Filter by region')
    parser.add_argument('--limit', type=int, default=10, help='Number of articles to show')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration and database
    config_path = Path(args.config)
    config = Config.load(config_path)
    database = Database(config.database_path)
    
    # Search with keyword combinations
    articles = search_combined_keywords(
        database, 
        args.keywords,
        require_all=args.require_all,
        ai_only=args.ai_only,
        region=args.region,
        limit=args.limit
    )
    
    # Display results
    if not articles:
        operator = "AND" if args.require_all else "OR"
        keywords_str = f" {operator} ".join(args.keywords)
        print(f"No articles found for: {keywords_str}")
        return
    
    operator = "AND" if args.require_all else "OR"
    keywords_str = f" {operator} ".join(args.keywords)
    region_text = f" in {args.region.upper()}" if args.region else ""
    
    print(f"\nFound {len(articles)} articles for: {keywords_str}{region_text}")
    print("-" * 80)
    
    for i, article in enumerate(articles, 1):
        relevance_indicator = "🤖" if article.ai_relevant else "  "
        region_indicator = f" [{article.region.upper()}]" if hasattr(article, 'region') else ""
        matching_keywords = ", ".join(article.matching_keywords) if hasattr(article, 'matching_keywords') else ""
        
        print(f"{i}. {relevance_indicator} {article.title}{region_indicator}")
        print(f"   Source: {article.source_name} | {article.published_at or 'Unknown date'}")
        print(f"   Matching keywords: {matching_keywords}")
        print(f"   URL: {article.url}")
        print()

if __name__ == '__main__':
    main()