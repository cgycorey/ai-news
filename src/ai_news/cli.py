"""Simple CLI interface for AI News using only standard library."""

import sys
import argparse
from pathlib import Path
from textwrap import fill
from datetime import datetime, timedelta
import json
import sqlite3

# Core imports (fast)
from .config import Config
from .database import Database
from .collector import SimpleCollector
from .search_collector import SearchEngineCollector
from .markdown_generator import MarkdownGenerator
from .entity_manager import get_entity_manager, Entity

# Heavy imports will be loaded lazily when needed
# from .entity_extractor import create_entity_extractor
# Academic imports removed - focusing on business intelligence
# from .intelligence_db import IntelligenceDB
# from .nlp_pipeline import NLPPipeline




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


def print_db_stats(db_stats, region_text=""):
    """Print database statistics."""
    print("\n" + "="*50)
    print(f"DATABASE STATISTICS{region_text}")
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


def print_entity_stats(stats):
    """Print entity statistics."""
    print("\n" + "="*50)
    print("ENTITY STATISTICS")
    print("="*50)
    print(f"Total entities:       {stats['total_entities']}")
    print(f"High confidence:      {stats['high_confidence_entities']}")
    print(f"Extraction patterns:  {stats['patterns_count']}")
    print(f"Exclusion patterns:   {stats['exclusion_patterns_count']}")
    
    print("\nEntities by type:")
    for entity_type, count in stats['entities_by_type'].items():
        print(f"  {entity_type}: {count}")    
    
    if stats['most_mentioned']:
        print("\nMost mentioned entities:")
        for name, count in stats['most_mentioned'][:5]:
            print(f"  {name}: {count} mentions")
    
    if stats['recently_discovered']:
        print("\nRecently discovered entities:")
        for name in stats['recently_discovered'][:5]:
            print(f"  {name}")
    
    print("="*50)


def print_entities(entities, show_details=False):
    """Print a list of entities."""
    if not entities:
        print("No entities found.")
        return
    
    print(f"\nFound {len(entities)} entities:")
    print("-" * 80)
    
    for i, entity in enumerate(entities, 1):
        print(f"{i}. {entity.name} ({entity.entity_type})")
        print(f"   Confidence: {entity.confidence:.2f} | Mentions: {entity.mention_count}")
        
        if show_details:
            if entity.description:
                print(f"   Description: {entity.description}")
            if entity.aliases:
                print(f"   Aliases: {', '.join(entity.aliases)}")
            if entity.last_seen:
                print(f"   Last seen: {entity.last_seen.strftime('%Y-%m-%d %H:%M')}")
        
        print()


def handle_cleanup_command(args, config, database):
    """Handle database cleanup operations."""
    print("\n" + "="*60)
    print("DATABASE CLEANUP")
    print("="*60)
    
    # Create backup if requested
    if args.backup:
        print("\nCreating database backup...")
        backup_result = database.backup_database(args.backup_path)
        if backup_result["success"]:
            print(f"✓ Backup created: {backup_result['backup_path']}")
            print(f"  Size: {backup_result['size_mb']} MB")
        else:
            print(f"✗ Backup failed: {backup_result.get('error', 'Unknown error')}")
            if not args.force:
                print("\nAborting cleanup due to backup failure.")
                return
    
    # Show preview if requested
    if args.preview:
        preview = database.get_cleanup_preview()
        print("\nCLEANUP PREVIEW:")
        print("-" * 40)
        print(f"Total articles: {preview['total_articles']}")
        if 'total_entities' in preview:
            print(f"Total entities: {preview['total_entities']}")
        print(f"Database size: {preview['database_size_mb']} MB")
        print("\nPotential cleanup items:")
        print(f"  Articles older than 90 days: {preview['articles_older_90_days']}")
        print(f"  Articles older than 180 days: {preview['articles_older_180_days']}")
        print(f"  Duplicate articles: {preview['duplicate_articles']}")
        print(f"  Empty articles: {preview['empty_articles']}")
        if 'orphaned_entities' in preview:
            print(f"  Orphaned entities: {preview['orphaned_entities']}")
        print("\n" + "="*60)
        return
    
    # If no specific operations requested, show preview and ask
    if not any([args.articles_older_than, args.remove_duplicates, 
                args.remove_empty, args.remove_non_ai, args.cleanup_entities, 
                args.optimize_only]):
        print("\nNo cleanup operations specified. Use --preview to see what can be cleaned.")
        print("\nAvailable cleanup options:")
        print("  --articles-older-than N    Remove articles older than N days")
        print("  --remove-duplicates        Remove duplicate articles")
        print("  --remove-empty            Remove articles with empty titles/summaries")
        print("  --remove-non-ai           Remove non-AI relevant articles")
        print("  --cleanup-entities        Remove orphaned entities")
        print("  --optimize-only           Run optimization only")
        print("  --preview                 Show cleanup preview")
        print("\n" + "="*60)
        return
    
    # Confirmation prompt
    if not args.force and not args.dry_run:
        print("\n⚠️  WARNING: This will permanently delete data from the database!")
        if args.backup:
            print("✓ Backup will be created first")
        else:
            print("✗ No backup will be created")
        
        response = input("\nContinue with cleanup? (yes/no): ").lower().strip()
        if response not in ['yes', 'y']:
            print("Cleanup cancelled.")
            return
    
    cleanup_summary = {
        "operations_performed": [],
        "total_articles_deleted": 0,
        "total_entities_deleted": 0
    }
    
    print("\nStarting cleanup operations...")
    print("-" * 40)
    
    try:
        # Remove old articles
        if args.articles_older_than:
            days = args.articles_older_than
            print(f"\nRemoving articles older than {days} days...")
            result = database.cleanup_old_articles(days, dry_run=args.dry_run)
            
            if result["articles_to_delete"] > 0:
                print(f"  Articles to delete: {result['articles_to_delete']}")
                print(f"  AI-relevant to delete: {result['ai_relevant_to_delete']}")
                print(f"  Sources affected: {result['sources_affected']}")
                
                if not args.dry_run:
                    print(f"  Articles deleted: {result.get('articles_deleted', 0)}")
                    cleanup_summary["total_articles_deleted"] += result.get('articles_deleted', 0)
                    cleanup_summary["operations_performed"].append(f"old_articles_{days}d")
            else:
                print("  No old articles found to delete")
        
        # Remove duplicate articles
        if args.remove_duplicates:
            print("\nRemoving duplicate articles...")
            result = database.remove_duplicate_articles(dry_run=args.dry_run)
            
            if result["articles_to_delete"] > 0:
                print(f"  Duplicate groups found: {result['duplicate_groups']}")
                print(f"  Articles to delete: {result['articles_to_delete']}")
                
                if not args.dry_run:
                    print(f"  Articles deleted: {result.get('articles_deleted', 0)}")
                    cleanup_summary["total_articles_deleted"] += result.get('articles_deleted', 0)
                    cleanup_summary["operations_performed"].append("duplicates")
            else:
                print("  No duplicate articles found")
        
        # Remove empty articles
        if args.remove_empty:
            print("\nRemoving empty articles...")
            result = database.remove_empty_articles(dry_run=args.dry_run)
            
            if result["articles_to_delete"] > 0:
                print(f"  Empty articles to delete: {result['articles_to_delete']}")
                
                if not args.dry_run:
                    print(f"  Articles deleted: {result.get('articles_deleted', 0)}")
                    cleanup_summary["total_articles_deleted"] += result.get('articles_deleted', 0)
                    cleanup_summary["operations_performed"].append("empty")
            else:
                print("  No empty articles found")
        
        # Remove non-AI articles
        if args.remove_non_ai:
            print("\nRemoving non-AI relevant articles...")
            with sqlite3.connect(database.db_path) as conn:
                count_result = conn.execute("""
                    SELECT COUNT(*) FROM articles WHERE ai_relevant = 0
                """).fetchone()
                articles_to_delete = count_result[0] if count_result else 0
                
                print(f"  Non-AI articles to delete: {articles_to_delete}")
                
                if not args.dry_run and articles_to_delete > 0:
                    conn.execute("DELETE FROM articles WHERE ai_relevant = 0")
                    deleted = conn.total_changes
                    print(f"  Articles deleted: {deleted}")
                    cleanup_summary["total_articles_deleted"] += deleted
                    cleanup_summary["operations_performed"].append("non_ai")
        
        # Clean up orphaned entities
        if args.cleanup_entities:
            print("\nRemoving orphaned entities...")
            result = database.remove_orphaned_entities(dry_run=args.dry_run)
            
            if result["total_orphaned"] > 0:
                print(f"  Orphaned entities to delete: {result['total_orphaned']}")
                print(f"  Low confidence entities: {result['low_confidence']}")
                
                for entity_type, count in result["by_type"].items():
                    print(f"    {entity_type}: {count}")
                
                if not args.dry_run:
                    print(f"  Entities deleted: {result.get('entities_deleted', 0)}")
                    cleanup_summary["total_entities_deleted"] += result.get('entities_deleted', 0)
                    cleanup_summary["operations_performed"].append("orphaned_entities")
            else:
                print("  No orphaned entities found")
        
        # Run optimization
        if args.optimize_only or any([args.articles_older_than, args.remove_duplicates, 
                                      args.remove_empty, args.cleanup_entities]):
            print("\nOptimizing database...")
            result = database.optimize_database()
            print(f"  Vacuum completed: {'✓' if result['vacuum_completed'] else '✗'}")
            print(f"  Analyze completed: {'✓' if result['analyze_completed'] else '✗'}")
            print(f"  Size before: {result['size_before_mb']} MB")
            print(f"  Size after: {result['size_after_mb']} MB")
            print(f"  Space saved: {result['space_saved_mb']} MB ({result['space_saved_percent']}%)")
            
            cleanup_summary["operations_performed"].append("optimization")
    
    except Exception as e:
        print(f"\n✗ Cleanup failed with error: {e}")
        return
    
    # Final summary
    print("\n" + "="*60)
    print("CLEANUP SUMMARY")
    print("="*60)
    
    if args.dry_run:
        print("DRY RUN MODE - No data was actually deleted")
    
    if cleanup_summary["operations_performed"]:
        print(f"Operations performed: {', '.join(cleanup_summary['operations_performed'])}")
        print(f"Total articles deleted: {cleanup_summary['total_articles_deleted']}")
        print(f"Total entities deleted: {cleanup_summary['total_entities_deleted']}")
    else:
        print("No cleanup operations were performed")
    
    if args.backup and 'backup_result' in locals() and backup_result["success"]:
        print(f"Backup saved to: {backup_result['backup_path']}")
    
    print("="*60)


def handle_schedule_command(args, config, database):
    """Handle schedule management commands."""
    
    if args.schedule_command == 'set':
        # Update schedule config
        config.schedule.enabled = True
        config.schedule.interval = args.interval
        
        # Save config
        config.save(config.config_path)
        
        print(f"✅ Schedule set to {args.interval}")
        print("\nTo enable automated collection, add this cron job:")
        print_schedule_cron_instruction(args.interval, config.config_path.parent)
        
    elif args.schedule_command == 'show':
        print_schedule_status(config)
        
    elif args.schedule_command == 'cron-setup':
        if config.schedule.enabled:
            print_schedule_cron_instruction(config.schedule.interval, config.config_path.parent)
        else:
            print("⚠️  No schedule configured. Use 'ai-news schedule set <interval>' first.")
            
    elif args.schedule_command == 'clear':
        config.schedule.enabled = False
        config.schedule.interval = "daily"
        config.schedule.last_collection = None
        config.schedule.next_collection = None
        
        config.save(config.config_path)
        print("✅ Schedule cleared. Remove any cron jobs you created.")
        
    else:
        print("❌ Unknown schedule command. Use --help to see available commands.")

def print_schedule_cron_instruction(interval, project_path):
    """Print cron setup instructions for given interval."""
    cron_commands = {
        'hourly': '0 * * * *',
        'daily': '0 2 * * *', 
        'weekly': '0 3 * * 0'
    }
    
    cron_time = cron_commands[interval]
    abs_path = Path(project_path).resolve()
    
    print(f"\n📅 Cron Setup Instructions:")
    print(f"1. Run: crontab -e")
    print(f"2. Add this line:")
    print(f"   {cron_time} cd {abs_path} && uv run ai-news collect --config {abs_path}/config.json")
    print(f"3. Save and exit")

def print_schedule_status(config):
    """Print current schedule status."""
    print("\n" + "="*50)
    print("COLLECTION SCHEDULE STATUS")
    print("="*50)
    
    if config.schedule.enabled:
        print(f"Status: ✅ ENABLED")
        print(f"Interval: {config.schedule.interval}")
        
        if config.schedule.last_collection:
            print(f"Last collection: {config.schedule.last_collection}")
        else:
            print("Last collection: Never")
            
        if config.schedule.next_collection:
            print(f"Next collection: {config.schedule.next_collection}")
            
        print("\nTo see cron setup instructions:")
        print("  ai-news schedule cron-setup")
    else:
        print("Status: ❌ DISABLED")
        print("No automated collection configured.")
        print("\nTo enable:")
        print("  ai-news schedule set <interval>")
    
    print("="*50)


def handle_feeds_command(args, config):
    """Handle feed management commands."""
    
    if args.feeds_command == 'add':
        # Create new feed
        ai_keywords = []
        if getattr(args, 'ai_keywords', None):
            ai_keywords = [kw.strip() for kw in args.ai_keywords.split(',')]
        
        from .config import FeedConfig, RegionConfig
        new_feed = FeedConfig(
            name=args.name,
            url=args.url,
            category=args.category,
            enabled=getattr(args, 'enabled', True),
            ai_keywords=ai_keywords
        )
        
        # Add to specified region
        region = getattr(args, 'region', 'global')
        if region not in config.regions:
            config.regions[region] = RegionConfig(name=region.title())
        
        config.regions[region].feeds.append(new_feed)
        config.save(config.config_path)
        
        print(f"✅ Added '{args.name}' to {region.upper()} region")
        
    elif args.feeds_command == 'list':
        region = getattr(args, 'region', None)
        
        if region:
            # List feeds for specific region
            if region in config.regions:
                feeds = config.regions[region].feeds
                print(f"\n📡 Feeds for {region.upper()} region:")
                print("-" * 50)
                
                for i, feed in enumerate(feeds, 1):
                    status = "ENABLED" if feed.enabled else "DISABLED"
                    print(f"{i}. [{status}] {feed.name} ({feed.category})")
                    print(f"   URL: {feed.url}")
                    print(f"   AI keywords: {len(feed.ai_keywords)}")
                    print()
            else:
                print(f"❌ No feeds found for region: {region}")
        else:
            # List all feeds by region
            print("\n🌍 All Feeds by Region:")
            print("=" * 50)
            
            for region_code, region_config in config.regions.items():
                if region_config.feeds:
                    print(f"\n{region_code.upper()} ({region_config.name}):")
                    for feed in region_config.feeds:
                        status = "✅" if feed.enabled else "❌"
                        print(f"  {status} {feed.name}")
    
    elif args.feeds_command == 'remove':
        region = getattr(args, 'region', 'global')
        feed_name = args.name
        
        if region in config.regions:
            feeds = config.regions[region].feeds
            original_count = len(feeds)
            
            # Remove feed by name
            config.regions[region].feeds = [
                feed for feed in feeds if feed.name != feed_name
            ]
            
            if len(config.regions[region].feeds) < original_count:
                config.save(config.config_path)
                print(f"✅ Removed '{feed_name}' from {region.upper()} region")
            else:
                print(f"❌ Feed '{feed_name}' not found in {region.upper()} region")
        else:
            print(f"❌ No feeds found for region: {region}")
    
    else:
        print("❌ Unknown feeds command. Use --help to see available commands.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='AI News Collector - Simple RSS-based news feeder')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--db', help='Override database path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect news from RSS feeds')
    collect_parser.add_argument('--region', choices=['us', 'uk', 'eu', 'apac', 'global'], help='Collect from specific region only')
    collect_parser.add_argument('--regions', help='Collect from multiple regions (comma-separated)')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List recent articles')
    list_parser.add_argument('--limit', type=int, default=20, help='Number of articles to show')
    list_parser.add_argument('--ai-only', action='store_true', help='Show only AI-relevant articles')
    list_parser.add_argument('--region', choices=['us', 'uk', 'eu', 'apac', 'global'], help='Filter by region')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search articles')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', type=int, default=20, help='Number of articles to show')
    search_parser.add_argument('--ai-only', action='store_true', help='Show only AI-relevant articles')
    search_parser.add_argument('--region', choices=['us', 'uk', 'eu', 'apac', 'global'], help='Filter by region')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    stats_parser.add_argument('--region', choices=['us', 'uk', 'eu', 'apac', 'global'], help='Show statistics for specific region')
    stats_parser.add_argument('--all-regions', action='store_true', help='Show statistics for all regions')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show current configuration')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Database cleanup and maintenance')
    cleanup_parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without actually deleting')
    cleanup_parser.add_argument('--backup', action='store_true', help='Create backup before cleanup')
    cleanup_parser.add_argument('--backup-path', help='Custom path for backup file')
    cleanup_parser.add_argument('--articles-older-than', type=int, metavar='DAYS', help='Remove articles older than specified days')
    cleanup_parser.add_argument('--remove-duplicates', action='store_true', help='Remove duplicate articles')
    cleanup_parser.add_argument('--remove-empty', action='store_true', help='Remove articles with empty titles/summaries')
    cleanup_parser.add_argument('--remove-non-ai', action='store_true', help='Remove non-AI relevant articles')
    cleanup_parser.add_argument('--cleanup-entities', action='store_true', help='Remove orphaned entities')
    cleanup_parser.add_argument('--optimize-only', action='store_true', help='Only run optimization (vacuum/analyze)')
    cleanup_parser.add_argument('--preview', action='store_true', help='Show cleanup preview')
    cleanup_parser.add_argument('--force', action='store_true', help='Skip confirmation prompts')
    
    # Entity management commands
    entity_parser = subparsers.add_parser('entities', help='Entity management commands')
    entity_subparsers = entity_parser.add_subparsers(dest='entity_command', help='Entity operations')
    
    # Entity list command
    entity_list_parser = entity_subparsers.add_parser('list', help='List entities')
    entity_list_parser.add_argument('--type', choices=['company', 'product', 'technology', 'person'], help='Filter by entity type')
    entity_list_parser.add_argument('--search', help='Search entities by name or description')
    entity_list_parser.add_argument('--limit', type=int, default=50, help='Number of entities to show')
    entity_list_parser.add_argument('--details', action='store_true', help='Show detailed information')
    
    # Entity stats command
    entity_stats_parser = entity_subparsers.add_parser('stats', help='Show entity statistics')
    
    # Entity extract command
    entity_extract_parser = entity_subparsers.add_parser('extract', help='Extract entities from text')
    entity_extract_parser.add_argument('text', help='Text to extract entities from')
    entity_extract_parser.add_argument('--discover', action='store_true', help='Discover new entities')
    
    # Entity add command
    entity_add_parser = entity_subparsers.add_parser('add', help='Add a new entity')
    entity_add_parser.add_argument('name', help='Entity name')
    entity_add_parser.add_argument('type', choices=['company', 'product', 'technology', 'person'], help='Entity type')
    entity_add_parser.add_argument('--description', help='Entity description')
    entity_add_parser.add_argument('--aliases', help='Comma-separated aliases')
    entity_add_parser.add_argument('--confidence', type=float, default=0.8, help='Initial confidence score')
    
    # Entity export command
    entity_export_parser = entity_subparsers.add_parser('export', help='Export entities to file')
    entity_export_parser.add_argument('file', help='Output file path')
    entity_export_parser.add_argument('--type', choices=['company', 'product', 'technology', 'person'], help='Export specific type only')
    
    # Entity import command
    entity_import_parser = entity_subparsers.add_parser('import', help='Import entities from file')
    entity_import_parser.add_argument('file', help='Input file path')
    
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
    
    # Enhanced NLP Pipeline commands
    nlp_parser = subparsers.add_parser('nlp', help='Advanced NLP processing commands')
    nlp_subparsers = nlp_parser.add_subparsers(dest='nlp_command', help='NLP operations')
    
    # NLP analyze command
    nlp_analyze_parser = nlp_subparsers.add_parser('analyze', help='Analyze text with enhanced NLP pipeline')
    nlp_analyze_parser.add_argument('text', help='Text to analyze')
    nlp_analyze_parser.add_argument('--title', help='Article title (for context)')
    nlp_analyze_parser.add_argument('--topics', action='store_true', help='Include topic modeling')
    nlp_analyze_parser.add_argument('--save', action='store_true', help='Save results to database')
    
    # NLP entities command
    nlp_entities_parser = nlp_subparsers.add_parser('entities', help='Extract entities with enhanced features')
    nlp_entities_parser.add_argument('text', help='Text to extract entities from')
    nlp_entities_parser.add_argument('--relationships', action='store_true', help='Extract entity relationships')
    nlp_entities_parser.add_argument('--disambiguate', action='store_true', help='Enable entity disambiguation')
    nlp_entities_parser.add_argument('--normalize', action='store_true', help='Normalize entity names')
    
    # NLP sentiment command
    nlp_sentiment_parser = nlp_subparsers.add_parser('sentiment', help='Analyze sentiment of text')
    nlp_sentiment_parser.add_argument('text', help='Text to analyze sentiment')
    nlp_sentiment_parser.add_argument('--detailed', action='store_true', help='Show detailed sentiment analysis')
    
    # NLP classify command
    nlp_classify_parser = nlp_subparsers.add_parser('classify', help='Classify text into categories')
    nlp_classify_parser.add_argument('text', help='Text to classify')
    nlp_classify_parser.add_argument('--relevance', action='store_true', help='Show AI relevance score')
    
    # Academic product idea commands removed
    # Focusing on business intelligence: entity tracking, trend analysis, market insights
    
    # Setup spaCy command
    setup_spacy_parser = subparsers.add_parser('setup-spacy', help='Download and setup spaCy models for NLP')
    
    # Setup NLTK command
    setup_nltk_parser = subparsers.add_parser('setup-nltk', help='Download and setup NLTK data for text processing')
    setup_nltk_parser.add_argument('--force', action='store_true', help='Force download even if already available')
    setup_nltk_parser.add_argument('--check', action='store_true', help='Only check NLTK data status')
    
    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Manage collection schedule')
    schedule_subparsers = schedule_parser.add_subparsers(dest='schedule_command', help='Schedule operations')
    
    # Schedule set command
    schedule_set_parser = schedule_subparsers.add_parser('set', help='Set collection schedule')
    schedule_set_parser.add_argument('interval', choices=['hourly', 'daily', 'weekly'], help='Collection interval')
    
    # Schedule show command
    schedule_show_parser = schedule_subparsers.add_parser('show', help='Show current schedule')
    
    # Schedule cron-setup command  
    schedule_cron_parser = schedule_subparsers.add_parser('cron-setup', help='Show cron setup instructions')
    
    # Schedule clear command
    schedule_clear_parser = subparsers.add_parser('clear', help='Clear schedule settings')
    
    # Feeds management commands
    feeds_parser = subparsers.add_parser('feeds', help='Manage RSS feeds')
    feeds_subparsers = feeds_parser.add_subparsers(dest='feeds_command', help='Feed operations')
    
    # Feeds add command
    feeds_add_parser = feeds_subparsers.add_parser('add', help='Add a new RSS feed')
    feeds_add_parser.add_argument('--name', required=True, help='Feed name')
    feeds_add_parser.add_argument('--url', required=True, help='Feed URL')
    feeds_add_parser.add_argument('--category', default='general', help='Feed category')
    feeds_add_parser.add_argument('--region', choices=['us', 'uk', 'eu', 'apac', 'global'], default='global', help='Target region')
    feeds_add_parser.add_argument('--enabled', action='store_true', default=True, help='Enable feed')
    feeds_add_parser.add_argument('--ai-keywords', help='Comma-separated AI keywords')
    
    # Feeds list command
    feeds_list_parser = feeds_subparsers.add_parser('list', help='List feeds')
    feeds_list_parser.add_argument('--region', choices=['us', 'uk', 'eu', 'apac', 'global'], help='List feeds for specific region')
    feeds_list_parser.add_argument('--enabled-only', action='store_true', help='Show only enabled feeds')
    
    # Feeds remove command
    feeds_remove_parser = feeds_subparsers.add_parser('remove', help='Remove a feed')
    feeds_remove_parser.add_argument('name', help='Feed name to remove')
    feeds_remove_parser.add_argument('--region', choices=['us', 'uk', 'eu', 'apac', 'global'], help='Region to remove from')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command provided, default to generating today's news
    if not args.command:
        print("🤖 No command specified - generating today's AI news digest...")
        print("Use --help to see all available commands.")
        print("💡 First-time setup: uv run ai-news setup-nltk\n")
        
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
            if getattr(args, 'region', None):
                # Collect from specific region
                collector = SimpleCollector(database)
                stats = collector.collect_region(config, args.region)
                print_stats(stats)
            elif getattr(args, 'regions', None):
                # Collect from multiple regions
                regions = [r.strip() for r in args.regions.split(',')]
                collector = SimpleCollector(database)
                stats = collector.collect_multiple_regions(config, regions)
                
                print(f"\n🌍 Multi-Region Collection Summary:")
                print(f"Regions processed: {stats['regions_processed']}")
                print(f"Total feeds processed: {stats['feeds_processed']}")
                print(f"Total articles fetched: {stats['total_fetched']}")
                print(f"Total articles added: {stats['total_added']}")
                print(f"Total AI-relevant added: {stats['ai_relevant_added']}")
            else:
                # Original behavior - collect from all regions
                collector = SimpleCollector(database)
                total_stats = {"feeds_processed": 0, "total_fetched": 0, "total_added": 0, "ai_relevant_added": 0}
                
                for region_code, region_config in config.regions.items():
                    if region_config.enabled:
                        region_stats = collector.collect_region(config, region_code)
                        total_stats["feeds_processed"] += region_stats["feeds_processed"]
                        total_stats["total_fetched"] += region_stats["total_fetched"]
                        total_stats["total_added"] += region_stats["total_added"]
                        total_stats["ai_relevant_added"] += region_stats["ai_relevant_added"]
                
                print_stats(total_stats)
            
        elif args.command == 'list':
            articles = database.get_articles(limit=args.limit, ai_only=args.ai_only, region=getattr(args, 'region', None))
            
            if not articles:
                print("No articles found.")
                return
            
            region_text = f" ({args.region.upper()})" if getattr(args, 'region', None) else ""
            print(f"\nShowing {len(articles)} recent articles{region_text}:")
            print("-" * 80)
            
            for i, article in enumerate(articles, 1):
                print_article_summary(article, i)
                
        elif args.command == 'search':
            all_articles = database.search_articles(args.query, limit=args.limit * 2, region=getattr(args, 'region', None))
            
            if args.ai_only:
                articles = [a for a in all_articles if a.ai_relevant][:args.limit]
            else:
                articles = all_articles[:args.limit]
            
            if not articles:
                print(f"No articles found for '{args.query}'.")
                return
            
            region_text = f" in {args.region.upper()}" if getattr(args, 'region', None) else ""
            print(f"\nFound {len(articles)} articles matching '{args.query}'{region_text}:")
            print("-" * 80)
            
            for i, article in enumerate(articles, 1):
                print_article_summary(article, i)
                
        elif args.command == 'stats':
            if getattr(args, 'all_regions', False):
                # Show stats for all regions
                print("\n" + "="*60)
                print("REGIONAL DATABASE STATISTICS")
                print("="*60)
                
                for region_code in ['us', 'uk', 'eu', 'apac', 'global']:
                    region_stats = database.get_stats(region=region_code)
                    if region_stats['total_articles'] > 0:
                        print(f"\n{region_code.upper()} REGION:")
                        print(f"  Total articles: {region_stats['total_articles']}")
                        print(f"  AI-relevant: {region_stats['ai_relevant_articles']}")
                        print(f"  Sources: {region_stats['sources_count']}")
            else:
                # Single region or global stats
                db_stats = database.get_stats(region=getattr(args, 'region', None))
                region_text = f" ({args.region.upper()})" if getattr(args, 'region', None) else ""
                print_db_stats(db_stats, region_text)
            
        elif args.command == 'config':
            print_config(config)
            
        elif args.command == 'cleanup':
            handle_cleanup_command(args, config, database)
            
        elif args.command == 'entities':
            entity_manager = get_entity_manager()
            
            if args.entity_command == 'list':
                if args.search:
                    entities = entity_manager.search_entities(args.search, args.type)
                else:
                    if args.type:
                        entities = entity_manager.get_entities_by_type(args.type)
                    else:
                        entities = list(entity_manager.entities.values())
                
                # Sort by mention count and confidence
                entities.sort(key=lambda x: (-x.mention_count, -x.confidence))
                entities = entities[:args.limit]
                
                print_entities(entities, show_details=args.details)
                
            elif args.entity_command == 'stats':
                stats = entity_manager.get_entity_statistics()
                print_entity_stats(stats)
                
                # Also show extraction statistics
                from .entity_extractor import create_entity_extractor
                extractor = create_entity_extractor()
                extraction_stats = extractor.get_extraction_statistics()
                print("\n" + "="*50)
                print("EXTRACTION STATISTICS")
                print("="*50)
                print(f"Total extractions: {extraction_stats['total_extractions']}")
                print(f"Average confidence: {extraction_stats['average_confidence']:.3f}")
                print("\nExtraction methods:")
                for method, count in extraction_stats['method_counts'].items():
                    print(f"  {method}: {count}")
                print("\nEntity types extracted:")
                for entity_type, count in extraction_stats['type_counts'].items():
                    print(f"  {entity_type}: {count}")
                print("="*50)
                
            elif args.entity_command == 'extract':
                from .entity_extractor import create_entity_extractor
                extractor = create_entity_extractor()
                
                print(f"Extracting entities from: {args.text[:100]}...\n")
                
                entities = extractor.extract_entities(args.text, discover_new=args.discover)
                
                if entities:
                    print(f"Found {len(entities)} entities:")
                    print("-" * 60)
                    for i, entity in enumerate(entities, 1):
                        print(f"{i}. {entity.text} ({entity.entity_type.value})")
                        print(f"   Confidence: {entity.confidence:.3f} | Method: {entity.extraction_method}")
                        if entity.canonical_name and entity.canonical_name != entity.text:
                            print(f"   Canonical: {entity.canonical_name}")
                        if entity.context:
                            print(f"   Context: {entity.context[:100]}...")
                        print()
                else:
                    print("No entities found.")
                    
            elif args.entity_command == 'add':
                aliases = [alias.strip() for alias in args.aliases.split(',')] if args.aliases else []
                
                new_entity = Entity(
                    name=args.name,
                    entity_type=args.type,
                    description=args.description or f"Manualy added {args.type}",
                    aliases=aliases,
                    confidence=args.confidence,
                    metadata={'source': 'manual_addition'}
                )
                
                if entity_manager.add_entity(new_entity, save_to_db=True):
                    print(f"✅ Successfully added entity: {args.name} ({args.type})")
                    if aliases:
                        print(f"   Aliases: {', '.join(aliases)}")
                    print(f"   Confidence: {args.confidence}")
                else:
                    print(f"❌ Failed to add entity: {args.name}")
                    
            elif args.entity_command == 'export':
                entity_manager.export_entities(args.file, args.type)
                print(f"✅ Entities exported to: {args.file}")
                
            elif args.entity_command == 'import':
                entity_manager.import_entities(args.file)
                print(f"✅ Entities imported from: {args.file}")
                
            else:
                print("❌ Unknown entity command. Use --help to see available commands.")
            
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
            
            # Auto-inference logic: if --topic is provided, default type to 'topic'
            if args.topic and args.type == 'daily':
                args.type = 'topic'
            
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
        
        elif args.command == 'nlp':
            cmd_nlp_processing(args)
        
        elif args.command == 'generate-ideas':
            print("❌ Academic feature removed: Product idea generation")
            print("✅ Focus on practical business intelligence:")
            print("   • Use 'ai-news list-entities' for company tracking")
            print("   • Use 'ai-news analyze' for trend analysis")
            print("   • Use 'ai-news search' for market insights")
            
        elif args.command == 'list-ideas':
            print("❌ Academic feature removed: Product idea listing")
            print("✅ Focus on practical business intelligence:")
            print("   • Use 'ai-news list-entities' for company tracking")
            print("   • Use 'ai-news analyze' for trend analysis")
            print("   • Use 'ai-news search' for market insights")
        
        elif args.command == 'setup-spacy':
            cmd_setup_spacy(args)
        
        elif args.command == 'setup-nltk':
            cmd_setup_nltk(args)
            
        elif args.command == 'schedule':
            handle_schedule_command(args, config, database)
            
        elif args.command == 'feeds':
            handle_feeds_command(args, config)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error executing command: {e}")
        sys.exit(1)


def cmd_nlp_processing(args):
    """Handle enhanced NLP processing commands."""
    try:
        print("🧠 Initializing Enhanced NLP Pipeline...")
        
        # Create simplified NLP pipeline
        from .nlp_pipeline import NLPPipeline
        pipeline = NLPPipeline(use_spacy=True)
        
        if args.nlp_command == 'analyze':
            print(f"📝 Analyzing text: {args.text[:100]}...\n")
            
            # Process text through simplified pipeline
            result = pipeline.process_text(args.text, args.title or "")
            
            # Display results
            print("\n" + "="*80)
            print("🔍 ENHANCED NLP ANALYSIS RESULTS")
            print("="*80)
            
            # Processing time
            print(f"⏱️  Processing time: {result.processing_time:.3f} seconds")
            
            # Sentiment analysis
            if result.sentiment:
                print(f"\n😊 SENTIMENT ANALYSIS:")
                print(f"   Label: {result.sentiment.label.value.upper()}")
                print(f"   Score: {result.sentiment.score:.3f} (-1.0 to 1.0)")
                print(f"   Confidence: {result.sentiment.confidence:.3f}")
            
            # Classification
            if result.classification:
                print(f"\n📂 TEXT CLASSIFICATION:")
                print(f"   Category: {result.classification.category.upper()}")
                print(f"   Confidence: {result.classification.confidence:.3f}")
                print(f"   AI Relevant: {'YES' if result.classification.ai_relevant else 'NO'}")
                if result.classification.keywords:
                    print(f"   Keywords: {', '.join(result.classification.keywords[:10])}")
            
            # Enhanced entity extraction
            if result.entities:
                print(f"\n🏢 ENHANCED ENTITY EXTRACTION:")
                print(f"   Total entities found: {len(result.entities)}")
                
                for i, entity in enumerate(result.entities, 1):
                    print(f"\n   {i}. {entity.text} ({entity.entity_type.value})")
                    print(f"      Confidence: {entity.confidence:.3f} | Method: {entity.extraction_method}")
                    
                    if entity.normalized_name:
                        print(f"      Normalized: {entity.normalized_name}")
                    
                    if entity.disambiguated:
                        print(f"      Disambiguated: ✓")
                    
                    if entity.linguistic_features:
                        print(f"      Linguistic features: {len(entity.linguistic_features)} extracted")
                    
                    if entity.relationships:
                        print(f"      Relationships: {len(entity.relationships)} found")
            
            # Entity relationships
            if result.entity_relationships:
                print(f"\n🔗 ENTITY RELATIONSHIPS:")
                print(f"   Total relationships: {len(result.entity_relationships)}")
                
                for i, rel in enumerate(result.entity_relationships, 1):
                    print(f"   {i}. {rel.source_entity} -> {rel.relationship_type} -> {rel.target_entity}")
                    print(f"      Confidence: {rel.confidence:.3f} | Evidence: {rel.evidence[:50]}...")
            
            # Topic modeling (if enabled)
            if result.topics:
                print(f"\n📊 TOPIC MODELING:")
                print(f"   Topics discovered: {len(result.topics)}")
                
                for i, topic in enumerate(result.topics, 1):
                    print(f"   {i}. Topic {topic.topic_id} (Coherence: {topic.coherence_score:.3f})")
                    top_words = [f"{word}({weight:.2f})" for word, weight in topic.topic_words[:5]]
                    print(f"      Top words: {', '.join(top_words)}")
            
            # Summary
            if result.summary:
                print(f"\n📋 AUTO-GENERATED SUMMARY:")
                print(f"   {result.summary}")
            
            # Save results if requested
            if args.save:
                save_nlp_results_to_database(result, database)
                print(f"\n💾 Results saved to database")
            
            print("="*80)
            
        elif args.nlp_command == 'entities':
            print(f"🏢 Extracting entities: {args.text[:100]}...\n")
            
            # Create enhanced entity extractor
            from .entity_extractor import create_entity_extractor
            extractor = create_entity_extractor(
                enable_relationships=args.relationships,
                enable_disambiguation=args.disambiguate
            )
            
            entities = extractor.extract_entities(args.text, include_context=True, discover_new=True)
            
            print(f"\n🎯 Enhanced Entity Extraction Results:")
            print(f"Total entities found: {len(entities)}")
            print("-" * 80)
            
            for i, entity in enumerate(entities, 1):
                print(f"\n{i}. {entity.text} ({entity.entity_type.value})")
                print(f"   Confidence: {entity.confidence:.3f} | Method: {entity.extraction_method}")
                
                if args.normalize and entity.normalized_name:
                    print(f"   Normalized: {entity.normalized_name}")
                
                if args.disambiguate and entity.disambiguated:
                    print(f"   Disambiguated: ✓")
                    if entity.metadata.get('disambiguation_evidence'):
                        print(f"   Evidence: {', '.join(entity.metadata['disambiguation_evidence'])}")
                
                if args.relationships and entity.relationships:
                    print(f"   Relationships: {len(entity.relationships)}")
                    for rel in entity.relationships:
                        print(f"      - {entity.text} -> {rel.relationship_type} -> {rel.target_entity}")
                
                if entity.linguistic_features:
                    print(f"   Linguistic features: {len(entity.linguistic_features)} extracted")
                    
                if entity.context:
                    print(f"   Context: {entity.context[:100]}...")
            
            print("-" * 80)
            
        elif args.nlp_command == 'sentiment':
            print(f"😊 Analyzing sentiment: {args.text[:100]}...\n")
            
            # Use pipeline's sentiment analyzer
            result = pipeline.process_text(args.text)
            
            if result.sentiment:
                sentiment = result.sentiment
                print(f"Sentiment: {sentiment.label.value.upper()}")
                print(f"Score: {sentiment.score:.3f} (-1.0 to 1.0)")
                print(f"Confidence: {sentiment.confidence:.3f}")
                
                if args.detailed:
                    print(f"\n📊 Detailed Analysis:")
                    print(f"Positive words: {len(sentiment.positive_words)}")
                    if sentiment.positive_words:
                        print(f"   {', '.join(sentiment.positive_words)}")
                    print(f"Negative words: {len(sentiment.negative_words)}")
                    if sentiment.negative_words:
                        print(f"   {', '.join(sentiment.negative_words)}")
                    print(f"Neutral score: {sentiment.neutral_score:.3f}")
            else:
                print("Could not analyze sentiment.")
        
        elif args.nlp_command == 'classify':
            print(f"📂 Classifying text: {args.text[:100]}...\n")
            
            # Use pipeline's text classifier
            result = pipeline.process_text(args.text)
            
            if result.classification:
                classification = result.classification
                print(f"Category: {classification.category.value.upper()}")
                print(f"Confidence: {classification.confidence:.3f}")
                
                if args.relevance:
                    print(f"\n🤖 AI Relevance:")
                    print(f"AI Relevant: {'YES' if classification.ai_relevant else 'NO'}")
                    print(f"Relevance Score: {classification.ai_relevance_score:.3f}")
                
                print(f"\n📋 Classification Probabilities:")
                for category, prob in classification.probabilities.items():
                    print(f"   {category}: {prob:.3f}")
                
                if classification.keywords:
                    print(f"\n🔑 Keywords: {', '.join(classification.keywords)}")
            else:
                print("Could not classify text.")
        
        else:
            print("❌ Unknown NLP command. Use --help to see available commands.")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"❌ Error in NLP processing: {e}")
        import traceback
        traceback.print_exc()


def save_nlp_results_to_database(result, database):
    """Save NLP analysis results to database."""
    # This would save the analysis results to the database
    # Implementation depends on database schema
    pass


def cmd_generate_ideas(args):
    """Removed: Academic product idea generation."""
    print("❌ Academic feature removed: Product idea generation")
    print("✅ Focus on practical business intelligence:")
    print("   • Entity tracking and mentions")
    print("   • Market trend analysis")
    print("   • Company monitoring")
    print("   • Technology trend identification")
    print("\n💡 Use 'ai-news list-entities' or 'ai-news analyze' instead!")
    return


def cmd_list_ideas(args):
    """List generated product ideas."""
    print("❌ Academic feature removed: Product idea listing")
    print("✅ Focus on practical business intelligence:")
    print("   • Use 'ai-news list-entities' for company tracking")
    print("   • Use 'ai-news analyze' for trend analysis")
    print("   • Use 'ai-news search' for market insights")
    return


def cmd_setup_spacy(args):
    """Setup spaCy models for NLP processing."""
    try:
        from .spacy_utils import setup_spacy_interactive
        
        print("🧠 Setting up spaCy models for advanced NLP...")
        print("This will download the required models for entity extraction and text processing.")
        print("")
        
        success = setup_spacy_interactive()
        
        if success:
            print("")
            print("✅ spaCy setup completed successfully!")
            print("You can now use advanced NLP features:")
            print("  • Entity extraction: uv run ai-news entities extract 'Your text here'")
            print("  • NLP analysis: uv run ai-news nlp analyze 'Your text here'")
            print("  • Enhanced news processing: uv run ai-news collect")
        else:
            print("")
            print("❌ spaCy setup failed or was cancelled.")
            print("You can try again later with: uv run ai-news setup-spacy")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"❌ Error during spaCy setup: {e}")
        sys.exit(1)


def cmd_setup_nltk(args):
    """Setup NLTK data for text processing."""
    try:
        from .nltk_utils import setup_nltk_data, check_nltk_data, get_nltk_info, get_missing_nltk_packages
        
        if args.check:
            # Only check NLTK data status
            print("📊 NLTK Data Status")
            print("=" * 40)
            
            status = check_nltk_data()
            missing = get_missing_nltk_packages()
            
            for package, available in status.items():
                status_icon = "✅" if available else "❌"
                print(f"{status_icon} {package}")
            
            print(f"\nPackages available: {len(status) - len(missing)}/{len(status)}")
            
            if missing:
                print(f"\n❌ Missing packages: {', '.join(missing)}")
                print("Run 'uv run ai-news setup-nltk' to download missing data.")
                sys.exit(1)
            else:
                print("\n✅ All NLTK data is available!")
                print("You can now use all text processing features.")
            return
        
        print("📚 Setting up NLTK data for text processing...")
        print("This will download required data for tokenization, stopwords, lemmatization, and more.")
        print("")
        
        # Show current status
        print("Current NLTK data status:")
        status = check_nltk_data()
        missing = get_missing_nltk_packages()
        
        for package, available in status.items():
            status_icon = "✅" if available else "❌"
            print(f"  {status_icon} {package}")
        
        if not missing and not args.force:
            print("\n✅ All NLTK data is already available!")
            print("No download needed.")
            return
        
        if args.force:
            print("\n⚠️  Force mode enabled - re-downloading all packages...")
        elif missing:
            print(f"\n📥 Missing packages to download: {', '.join(missing)}")
        
        print("")
        
        # Setup NLTK data
        success = setup_nltk_data(force=args.force, show_progress=True)
        
        if success:
            print("")
            print("✅ NLTK setup completed successfully!")
            print("")
            print("🎉 Text processing features are now ready:")
            print("  • Sentence tokenization: Splitting text into sentences")
            print("  • Word tokenization: Breaking text into words")
            print("  • Stopword filtering: Removing common words")
            print("  • Lemmatization: Converting words to base forms")
            print("  • POS tagging: Identifying parts of speech")
            print("  • Keyword extraction: Finding important terms")
            print("")
            print("You can now use text processing commands:")
            print("  • Collect news: uv run ai-news collect")
            print("  • Search articles: uv run ai-news search 'query'")
            print("  • Generate digests: uv run ai-news digest")
            print("  • NLP analysis: uv run ai-news nlp analyze 'Your text'")
        else:
            print("")
            print("❌ NLTK setup failed for some packages.")
            print("")
            print("🔧 Troubleshooting tips:")
            print("  • Check your internet connection")
            print("  • Try running: uv run ai-news setup-nltk --force")
            print("  • Check NLTK data status: uv run ai-news setup-nltk --check")
            print("  • Clear NLTK cache and retry: rm ~/.ai_news_cache/nltk_status.json")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user.")
        print("Setup was not completed. You can try again later.")
    except ImportError as e:
        print(f"❌ NLTK utilities not available: {e}")
        print("Make sure the nltk_utils module is properly installed.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during NLTK setup: {e}")
        print("")
        print("🔧 Troubleshooting tips:")
        print("  • Check your internet connection")
        print("  • Try running: uv run ai-news setup-nltk --force")
        print("  • Check NLTK data status: uv run ai-news setup-nltk --check")
        sys.exit(1)


if __name__ == '__main__':
    main()