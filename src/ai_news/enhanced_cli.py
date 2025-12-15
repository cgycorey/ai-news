"""Enhanced CLI interface for multi-keyword AI News features."""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from .config import Config
from .database import Database
from .enhanced_collector import (
    EnhancedCollector,
    EnhancedMultiKeywordCollector,
    KeywordCategory,
    MultiKeywordResult
)

def format_enhanced_results(results: List[tuple], show_details: bool = False):
    """Format enhanced search results for display."""
    if not results:
        print("🔍 No articles found matching your criteria.")
        return
    
    print(f"\n🔍 Found {len(results)} matching articles:")
    print("=" * 80)
    
    for i, (article, result) in enumerate(results, 1):
        # Article header
        relevance_indicator = "🤖" if article.ai_relevant else "  "
        print(f"{i}. {relevance_indicator} {article.title}")
        
        # Article metadata
        date_str = article.published_at.strftime("%Y-%m-%d") if article.published_at else "Unknown"
        print(f"   📅 {date_str} | 📰 {article.source_name} | 🌍 {article.region.upper()}")
        
        # Enhanced scores
        print(f"   📊 Final Score: {result.final_score:.3f}")
        print(f"   🎯 Total Score: {result.total_score:.3f} | Intersection: {result.intersection_score:.3f} | Region Boost: {result.region_boost:.3f}")
        
        # Category scores
        if result.category_scores:
            categories_text = ", ".join([f"{cat}: {score:.2f}" for cat, score in result.category_scores.items()])
            print(f"   📈 Categories: {categories_text}")
        
        # Top keyword matches
        if show_details and result.matches:
            print(f"   🔍 Top matches:")
            for match in result.matches[:3]:
                print(f"      • {match.keyword} ({match.category}): {match.score:.3f}")
                if len(match.context) > 60:
                    context = match.context[:60] + "..."
                else:
                    context = match.context
                print(f"        Context: {context}")
        
        # Content snippet
        snippet = article.summary or article.content[:150]
        if len(snippet) > 150:
            snippet = snippet[:150] + "..."
        print(f"   📄 {snippet}")
        
        print(f"   🔗 {article.url}")
        print()


def run_enhanced_search(
    database: Database,
    query_parts: List[str],
    region: str = "global",
    min_score: float = 0.1,
    limit: int = 20,
    show_details: bool = False
):
    """Run enhanced multi-keyword search."""
    
    # Initialize enhanced collector
    enhanced_collector = EnhancedMultiKeywordCollector(performance_mode=True)
    
    # Build keyword categories from query parts
    categories = {}
    weights = {}
    
    # Map query parts to categories
    category_mapping = {
        'ai': enhanced_collector.categories['ai'].keywords,
        'ml': ['ML', 'machine learning', 'deep learning', 'neural network', 'algorithm'],
        'insurance': enhanced_collector.categories['insurance'].keywords,
        'healthcare': enhanced_collector.categories['healthcare'].keywords,
        'fintech': enhanced_collector.categories['fintech'].keywords
    }
    
    required_combinations = []
    
    for part in query_parts:
        part_lower = part.lower()
        if part_lower in category_mapping:
            categories[part_lower] = category_mapping[part_lower]
            weights[part_lower] = 1.0
            required_combinations.append([part_lower])
    
    if not categories:
        print("❌ No valid keyword categories found in query.")
        print("💡 Try: 'ai insurance', 'ml healthcare', 'ai fintech', etc.")
        return
    
    print(f"🔍 Searching for: {' + '.join(query_parts)}")
    print(f"🌍 Region: {region.upper()}")
    print(f"📊 Minimum score: {min_score}")
    print()
    
    # Get articles from database
    articles = database.get_articles(limit=1000, region=region if region != 'global' else None)
    
    # Filter articles using enhanced analysis
    filtered_results = []
    
    for article in articles:
        result = enhanced_collector.analyze_multi_keywords(
            title=article.title,
            content=article.content,
            categories=categories,
            region=region,
            min_score=min_score
        )
        
        # Check if article meets required combinations
        if result.is_relevant:
            matched_categories = set(result.category_scores.keys())
            
            # Check if we have matches for the requested categories
            category_matches = 0
            for combo in required_combinations:
                if set(combo).issubset(matched_categories):
                    category_matches += 1
            
            if category_matches > 0:
                filtered_results.append((article, result))
    
    # Sort by final score
    filtered_results.sort(key=lambda x: x[1].final_score, reverse=True)
    
    # Display results
    format_enhanced_results(filtered_results[:limit], show_details)
    
    # Generate coverage report
    if filtered_results:
        print("\n" + "=" * 50)
        print("COVERAGE REPORT")
        print("=" * 50)
        
        # Category statistics
        category_stats = {}
        for _, result in filtered_results:
            for category, score in result.category_scores.items():
                if category not in category_stats:
                    category_stats[category] = {'count': 0, 'total_score': 0}
                category_stats[category]['count'] += 1
                category_stats[category]['total_score'] += score
        
        for category, stats in category_stats.items():
            avg_score = stats['total_score'] / stats['count']
            print(f"{category.upper()}: {stats['count']} articles (avg score: {avg_score:.3f})")
        
        # Intersection statistics
        intersections = {}
        for _, result in filtered_results:
            if result.intersection_score > 0:
                categories_key = " + ".join(sorted(result.category_scores.keys()))
                if categories_key not in intersections:
                    intersections[categories_key] = 0
                intersections[categories_key] += 1
        
        if intersections:
            print("\nINTERSECTIONS:")
            for intersection, count in intersections.items():
                print(f"  {intersection}: {count} articles")
        
        print(f"\nTotal matching articles: {len(filtered_results)}")
        avg_score = sum(r.final_score for _, r in filtered_results) / len(filtered_results)
        print(f"Average score: {avg_score:.3f}")


def run_intersection_search(
    database: Database,
    categories: List[str],
    region: str = "global",
    limit: int = 20,
    show_details: bool = False
):
    """Run intersection search for articles matching ALL specified categories."""
    
    enhanced_collector = EnhancedMultiKeywordCollector(performance_mode=True)
    
    # Map categories to keywords
    category_mapping = {
        'ai': enhanced_collector.categories['ai'].keywords,
        'ml': ['ML', 'machine learning', 'deep learning', 'neural network', 'algorithm'],
        'insurance': enhanced_collector.categories['insurance'].keywords,
        'healthcare': enhanced_collector.categories['healthcare'].keywords,
        'fintech': enhanced_collector.categories['fintech'].keywords
    }
    
    # Build category dictionary
    search_categories = {}
    for category in categories:
        if category.lower() in category_mapping:
            search_categories[category.lower()] = category_mapping[category.lower()]
    
    if len(search_categories) < 2:
        print("❌ Intersection search requires at least 2 categories.")
        return
    
    print(f"🔍 Intersection search for: {' AND '.join(categories)}")
    print(f"🌍 Region: {region.upper()}")
    print()
    
    # Get articles
    articles = database.get_articles(limit=1000, region=region if region != 'global' else None)
    
    # Filter for intersection matches
    intersection_results = []
    
    for article in articles:
        result = enhanced_collector.analyze_multi_keywords(
            title=article.title,
            content=article.content,
            categories=search_categories,
            region=region,
            min_score=0.1
        )
        
        # Check if article matches ALL requested categories
        matched_categories = set(result.category_scores.keys())
        if set(search_categories.keys()).issubset(matched_categories):
            intersection_results.append((article, result))
    
    # Sort by intersection score (prioritize multi-category matches)
    intersection_results.sort(key=lambda x: x[1].intersection_score, reverse=True)
    
    print(f"✅ Found {len(intersection_results)} articles matching ALL categories: {' + '.join(categories)}")
    print()
    
    # Display results
    format_enhanced_results(intersection_results[:limit], show_details)


def demo_combinations(database: Database):
    """Demonstrate various multi-keyword combinations."""
    
    print("🎯 Multi-Keyword Combination Demo")
    print("=" * 50)
    
    demo_queries = [
        (['ai', 'insurance'], 'uk', "AI + Insurance in UK"),
        (['ai', 'healthcare'], 'us', "AI + Healthcare in US"),
        (['ml', 'fintech'], 'eu', "ML + FinTech in EU"),
        (['ai', 'insurance'], 'global', "AI + Insurance (Global)")
    ]
    
    for categories, region, description in demo_queries:
        print(f"\n{description}")
        print("-" * len(description))
        
        enhanced_collector = EnhancedMultiKeywordCollector(performance_mode=True)
        
        # Map categories
        category_mapping = {
            'ai': enhanced_collector.categories['ai'].keywords,
            'ml': ['ML', 'machine learning', 'deep learning'],
            'insurance': enhanced_collector.categories['insurance'].keywords,
            'healthcare': enhanced_collector.categories['healthcare'].keywords,
            'fintech': enhanced_collector.categories['fintech'].keywords
        }
        
        search_categories = {}
        for cat in categories:
            if cat in category_mapping:
                search_categories[cat] = category_mapping[cat]
        
        if not search_categories:
            print("  ❌ No valid categories")
            continue
        
        # Get limited sample of articles
        articles = database.get_articles(limit=200, region=region if region != 'global' else None)
        
        # Analyze
        matches = 0
        total_score = 0
        intersection_matches = 0
        
        for article in articles:
            result = enhanced_collector.analyze_multi_keywords(
                title=article.title,
                content=article.content,
                categories=search_categories,
                region=region,
                min_score=0.05
            )
            
            if result.is_relevant:
                matches += 1
                total_score += result.final_score
                if result.intersection_score > 0:
                    intersection_matches += 1
        
        print(f"  Articles analyzed: {len(articles)}")
        if len(articles) > 0:
            print(f"  Matches found: {matches} ({matches/len(articles)*100:.1f}%)")
        else:
            print(f"  Matches found: {matches} (0.0%)")
        print(f"  Intersection matches: {intersection_matches}")
        
        if matches > 0:
            print(f"  Average score: {total_score/matches:.3f}")
        
        if matches > 0:
            print(f"  📊 Coverage: {matches} articles found")
        else:
            print(f"  ⚠️  No matches found")


def main():
    """Enhanced CLI main entry point."""
    parser = argparse.ArgumentParser(description='Enhanced AI News Multi-Keyword Search')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Enhanced commands')
    
    # Multi-keyword search command
    multi_parser = subparsers.add_parser('multi', help='Multi-keyword search')
    multi_parser.add_argument('keywords', nargs='+', help='Keywords (e.g., ai insurance healthcare)')
    multi_parser.add_argument('--region', choices=['us', 'uk', 'eu', 'apac', 'global'], 
                             default='global', help='Filter by region')
    multi_parser.add_argument('--min-score', type=float, default=0.1, help='Minimum relevance score')
    multi_parser.add_argument('--limit', type=int, default=20, help='Number of articles to show')
    multi_parser.add_argument('--details', action='store_true', help='Show detailed match information')
    
    # Intersection search command
    intersection_parser = subparsers.add_parser('intersection', help='Intersection search (ALL categories must match)')
    intersection_parser.add_argument('categories', nargs='+', help='Categories that must all match')
    intersection_parser.add_argument('--region', choices=['us', 'uk', 'eu', 'apac', 'global'], 
                                    default='global', help='Filter by region')
    intersection_parser.add_argument('--limit', type=int, default=20, help='Number of articles to show')
    intersection_parser.add_argument('--details', action='store_true', help='Show detailed match information')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Demonstrate multi-keyword combinations')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load configuration and database
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("💡 Run 'python -m ai_news.cli init' to create a configuration file.")
        sys.exit(1)
    
    try:
        config = Config.load(config_path)
        database = Database(config.database_path)
    except Exception as e:
        print(f"❌ Error loading configuration or database: {e}")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'multi':
            run_enhanced_search(
                database=database,
                query_parts=args.keywords,
                region=args.region,
                min_score=args.min_score,
                limit=args.limit,
                show_details=args.details
            )
        
        elif args.command == 'intersection':
            run_intersection_search(
                database=database,
                categories=args.categories,
                region=args.region,
                limit=args.limit,
                show_details=args.details
            )
        
        elif args.command == 'demo':
            demo_combinations(database)
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user.")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()