"""Simple CLI interface for AI News using only standard library."""

import sys
import argparse
from pathlib import Path
from textwrap import fill
from datetime import datetime, timedelta

from .config import Config
from .database import Database
from .collector import SimpleCollector
from .search_collector import SearchEngineCollector
from .markdown_generator import MarkdownGenerator


def print_article_summary(article, index=1):
    """Print a formatted article summary."""
    print(f"\n{index}. {article.title}")
    print(f"   Source: {article.source_name} | {article.published_at.strftime('%Y-%m-%d') if article.published_at else 'Unknown'}")
    ai_marker = "✓" if article.ai_relevant else "✗"
    print(f"   AI Relevant: {ai_marker} | Category: {article.category}")
    
    # Truncate summary
    summary = article.summary[:100] + "..." if len(article.summary) > 100 else article.summary
    print(f"   {summary}")
    print(f"   URL: {article.url}")


def print_stats(stats):
    """Print formatted statistics."""
    print("\n" + "="*50)
    print("COLLECTION SUMMARY")
    print("="*50)
    print(f"Feeds processed: {stats['feeds_processed']}")
    print(f"Articles fetched: {stats['total_fetched']}")
    print(f"Articles added:   {stats['total_added']}")
    print(f"AI-relevant added: {stats['ai_relevant_added']}")
    print("="*50)


def print_db_stats(db_stats):
    """Print database statistics."""
    print("\n" + "="*50)
    print("DATABASE STATISTICS")
    print("="*50)
    print(f"Total articles:      {db_stats['total_articles']}")
    print(f"AI-relevant articles: {db_stats['ai_relevant_articles']}")
    print(f"Sources:             {db_stats['sources_count']}")
    print(f"AI relevance rate:   {db_stats['ai_relevance_rate']}")
    print("="*50)


def print_config(config):
    """Print current configuration."""
    print("\n" + "="*50)
    print("CONFIGURATION")
    print("="*50)
    print(f"Database path:        {config.database_path}")
    print(f"Max articles per feed: {config.max_articles_per_feed}")
    print(f"Collection interval:  {config.collection_interval_hours} hours")
    print(f"Configured feeds:     {len(config.feeds)}")
    print("\nFeeds:")
    for i, feed in enumerate(config.feeds, 1):
        status = "ENABLED" if feed.enabled else "DISABLED"
        print(f"  {i}. [{status}] {feed.name} ({feed.category})")
        print(f"     URL: {feed.url}")
        print(f"     AI keywords: {len(feed.ai_keywords)}")
    print("="*50)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='AI News Collector - Simple RSS-based news feeder')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--db', help='Override database path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect news from RSS feeds')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List recent articles')
    list_parser.add_argument('--limit', type=int, default=20, help='Number of articles to show')
    list_parser.add_argument('--ai-only', action='store_true', help='Show only AI-relevant articles')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search articles')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', type=int, default=20, help='Number of articles to show')
    search_parser.add_argument('--ai-only', action='store_true', help='Show only AI-relevant articles')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show current configuration')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show full article details')
    show_parser.add_argument('article_id', type=int, help='Article ID to display')
    
    # Search command for web search
    search_parser = subparsers.add_parser('websearch', help='Search web for AI + topic articles')
    search_parser.add_argument('topic', help='Topic to search for with AI')
    search_parser.add_argument('--limit', type=int, default=10, help='Number of articles to add')
    search_parser.add_argument('--trending', action='store_true', help='Search trending AI topics')
    
    # Digest commands
    digest_parser = subparsers.add_parser('digest', help='Generate news digests')
    digest_parser.add_argument('--type', choices=['daily', 'weekly', 'topic'], default='daily', help='Type of digest')
    digest_parser.add_argument('--date', help='Date for daily digest (YYYY-MM-DD)')
    digest_parser.add_argument('--days', type=int, default=7, help='Days for topic analysis')
    digest_parser.add_argument('--topic', help='Topic for analysis (required for topic digest)')
    digest_parser.add_argument('--ai-only', action='store_true', help='Include only AI-relevant articles')
    digest_parser.add_argument('--save', action='store_true', help='Save digest to file')
    digest_parser.add_argument('--output', default='digests', help='Output directory for saved digests')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command provided, default to generating today's news
    if not args.command:
        print("🤖 No command specified - generating today's AI news digest...")
        print("Use --help to see all available commands.\n")
        
        # Create a simple object with digest defaults
        class DigestArgs:
            def __init__(self):
                self.command = 'digest'
                self.type = 'daily'
                self.ai_only = True
                self.save = False
                self.output = 'digests'
                self.date = None
                self.days = 7
                self.topic = None
                self.config = args.config if hasattr(args, 'config') else 'config.json'
                self.db = getattr(args, 'db', None)
        
        args = DigestArgs()
    
    # Load configuration
    config_path = Path(args.config)
    try:
        config = Config.load(config_path)
        db_path = args.db or config.database_path
        database = Database(db_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Execute command
    try:
        if args.command == 'collect':
            print("Starting news collection...")
            collector = SimpleCollector(database)
            stats = collector.collect_all_feeds(config.feeds)
            print_stats(stats)
            
        elif args.command == 'list':
            articles = database.get_articles(limit=args.limit, ai_only=args.ai_only)
            
            if not articles:
                print("No articles found.")
                return
            
            print(f"\nShowing {len(articles)} recent articles:")
            print("-" * 80)
            
            for i, article in enumerate(articles, 1):
                print_article_summary(article, i)
                
        elif args.command == 'search':
            all_articles = database.search_articles(args.query, limit=args.limit * 2)  # Get more to filter
            
            if args.ai_only:
                articles = [a for a in all_articles if a.ai_relevant][:args.limit]
            else:
                articles = all_articles[:args.limit]
            
            if not articles:
                print(f"No articles found for '{args.query}'.")
                return
            
            print(f"\nFound {len(articles)} articles matching '{args.query}':")
            print("-" * 80)
            
            for i, article in enumerate(articles, 1):
                print_article_summary(article, i)
                
        elif args.command == 'stats':
            db_stats = database.get_stats()
            print_db_stats(db_stats)
            
        elif args.command == 'config':
            print_config(config)
            
        elif args.command == 'show':
            # Get all articles and find the one with matching ID
            articles = database.get_articles(limit=1000)
            article = next((a for a in articles if a.id == args.article_id), None)
            
            if not article:
                print(f"Article with ID {args.article_id} not found.")
                return
            
            print("\n" + "="*80)
            print(f"ARTICLE DETAILS: {article.title}")
            print("="*80)
            print(f"Source:     {article.source_name}")
            print(f"Author:     {article.author or 'Unknown'}")
            print(f"Published:  {article.published_at.strftime('%Y-%m-%d %H:%M') if article.published_at else 'Unknown'}")
            print(f"Category:   {article.category}")
            print(f"AI Relevant: {'Yes' if article.ai_relevant else 'No'}")
            if article.ai_keywords_found:
                print(f"AI Keywords: {', '.join(article.ai_keywords_found)}")
            print(f"URL:        {article.url}")
            print("\nSUMMARY:")
            print(fill(article.summary, width=80))
            if len(article.content) > len(article.summary):
                print(f"\nFULL CONTENT:")
                print(fill(article.content, width=80))
            print("="*80)
            
        elif args.command == 'digest':
            
            md_gen = MarkdownGenerator(database)
            
            if args.type == 'daily':
                if args.date:
                    try:
                        digest_date = datetime.strptime(args.date, '%Y-%m-%d')
                    except ValueError:
                        print("Error: Date must be in YYYY-MM-DD format")
                        return
                else:
                    digest_date = datetime.now()
                
                print(f"Generating daily digest for {digest_date.strftime('%Y-%m-%d')}...")
                content = md_gen.generate_daily_digest(digest_date, args.ai_only)
                
            elif args.type == 'weekly':
                if args.date:
                    try:
                        start_date = datetime.strptime(args.date, '%Y-%m-%d')
                    except ValueError:
                        print("Error: Date must be in YYYY-MM-DD format")
                        return
                else:
                    start_date = datetime.now() - timedelta(days=7)
                
                print(f"Generating weekly digest starting {start_date.strftime('%Y-%m-%d')}...")
                content = md_gen.generate_weekly_digest(start_date)
                
            elif args.type == 'topic':
                if not args.topic:
                    print("Error: Topic is required for topic analysis")
                    return
                
                print(f"Generating topic analysis for '{args.topic}' (last {args.days} days)...")
                content = md_gen.generate_topic_analysis(args.topic, args.days)
            
            # Display or save the digest
            if args.save:
                filename = f"{args.type}_digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                file_path = md_gen.save_digest_to_file(content, filename, args.output)
                print(f"Digest saved to: {file_path}")
            else:
                print("\n" + content)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error executing command: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()