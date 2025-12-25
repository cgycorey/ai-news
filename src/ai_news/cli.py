"""Simple CLI interface for AI News using only standard library."""

import sys
import argparse
from pathlib import Path
from textwrap import fill
from datetime import datetime, timedelta
import json
import sqlite3
from typing import List

# Core imports (fast)
from .config import Config
from .database import Database
from .collector import SimpleCollector
from .search_collector import SearchEngineCollector
from .markdown_generator import MarkdownGenerator
from .entity_manager import get_entity_manager, Entity
from .topic_discovery import create_topic_discovery

# Heavy imports will be loaded lazily when needed
# from .entity_extractor import create_entity_extractor
# Academic imports removed - focusing on business intelligence
# from .intelligence_db import IntelligenceDB
# from .nlp_pipeline import NLPPipeline

# Enhanced multi-keyword functionality - lazy loaded for performance
# from .enhanced_collector import EnhancedMultiKeywordCollector, KeywordCategory




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
        enabled_only = getattr(args, 'enabled_only', False)
        
        if region:
            # List feeds for specific region
            if region in config.regions:
                feeds = config.regions[region].feeds
                if enabled_only:
                    feeds = [feed for feed in feeds if feed.enabled]
                
                print(f"\n📡 Feeds for {region.upper()} region:")
                if enabled_only:
                    print("(Enabled feeds only)")
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
            if enabled_only:
                print("(Enabled feeds only)")
            print("=" * 50)
            
            for region_code, region_config in config.regions.items():
                feeds = region_config.feeds
                if enabled_only:
                    feeds = [feed for feed in feeds if feed.enabled]
                
                if feeds:
                    print(f"\n{region_code.upper()} ({region_config.name}):")
                    for feed in feeds:
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

    # Topic management commands
    topic_parser = subparsers.add_parser('topics', help='Topic management and discovery')
    topic_subparsers = topic_parser.add_subparsers(dest='topic_command', help='Topic operations')

    # Topics list command
    topic_list_parser = topic_subparsers.add_parser('list', help='List all topics')
    topic_list_parser.add_argument('--verbose', action='store_true', help='Show detailed information')

    # Topics add command
    topic_add_parser = topic_subparsers.add_parser('add', help='Add a new topic')
    topic_add_parser.add_argument('name', help='Topic name')
    topic_add_parser.add_argument('keywords', nargs='+', help='Keywords for this topic')
    topic_add_parser.add_argument('--no-discover', action='store_true', help='Disable auto-discovery for this topic')

    # Topics remove command
    topic_remove_parser = topic_subparsers.add_parser('remove', help='Remove a topic')
    topic_remove_parser.add_argument('name', help='Topic name to remove')

    # Topics discover command
    topic_discover_parser = topic_subparsers.add_parser('discover', help='Run topic discovery on database')
    topic_discover_parser.add_argument('topic', help='Topic name to discover for (or "all" for all topics)')
    topic_discover_parser.add_argument('--min-occurrence', type=int, default=3, help='Minimum occurrence threshold')
    topic_discover_parser.add_argument('--prune', action='store_true', help='Prune stale discoveries after running')
    topic_discover_parser.add_argument('--use-spacy', action='store_true', default=True, help='Use spaCy for term extraction (default: True)')
    topic_discover_parser.add_argument('--no-spacy', action='store_true', help='Disable spaCy, use basic extraction')
    topic_discover_parser.add_argument('--min-relevance', type=float, default=0.3, help='Minimum domain relevance score (default: 0.3)')

    # Topics stats command
    topic_stats_parser = topic_subparsers.add_parser('stats', help='Show discovery statistics')
    topic_stats_parser.add_argument('topic', help='Topic name (or "all" for all topics)')

    # Topics suggest command
    topic_suggest_parser = topic_subparsers.add_parser('suggest', help='Suggest related topics')
    topic_suggest_parser.add_argument('topic', help='Topic name to analyze')

    # Show command
    show_parser = subparsers.add_parser('show', help='Show full article details')
    show_parser.add_argument('article_id', type=int, help='Article ID to display')
    
    # Search command for web search
    search_parser = subparsers.add_parser('websearch', help='Search web for AI + topic articles with intersection detection')
    search_parser.add_argument(
        'topics', 
        nargs='+',
        help='One or more topics to search for with AI (e.g., "healthcare" or "healthcare" "finance")'
    )
    search_parser.add_argument('--limit', type=int, default=10, help='Max articles per topic')
    search_parser.add_argument('--min-confidence', type=float, default=0.25,
                              help='Minimum confidence for intersection detection (default: 0.25)')
    search_parser.add_argument('--max-intersection-size', type=int, default=3,
                              help='Maximum number of topics in an intersection (default: 3)')
    search_parser.add_argument('--no-intersections', action='store_true',
                              help='Skip intersection detection (individual topics only)')
    search_parser.add_argument('--include-rss', action='store_true',
                              help='Also collect articles from RSS feeds during websearch')
    search_parser.add_argument('--regions', default='global',
                              help='Regions to collect RSS feeds from (default: global). Comma-separated: us,uk,eu,apac,global')
    search_parser.add_argument('--save', action='store_true',
                              help='Automatically save articles without prompting')
    search_parser.add_argument('--trending', action='store_true', help='Search trending AI topics')
    

    
    # Digest commands
    digest_parser = subparsers.add_parser('digest', help='Generate news digests')
    digest_parser.add_argument('--type', choices=['daily', 'weekly', 'topic'], default='daily', help='Type of digest')
    digest_parser.add_argument('--date', help='Date for daily digest (YYYY-MM-DD)')
    digest_parser.add_argument('--days', type=int, default=7, help='Days for topic analysis')
    digest_parser.add_argument('--topic', nargs='+', help='One or more topics for analysis (required for topic digest)')
    digest_parser.add_argument('--ai-only', action='store_true', help='Include only AI-relevant articles')
    digest_parser.add_argument('--save', action='store_true', help='Save digest to file')
    digest_parser.add_argument('--output', default='digests', help='Output directory for saved digests')
    digest_parser.add_argument('--no-spacy', action='store_true', help='Disable spaCy analysis, use keyword-only mode')
    
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
    schedule_clear_parser = schedule_subparsers.add_parser('clear', help='Clear schedule settings')
    
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
    
    # Feed discovery commands for automatic RSS feed finding
    add_topic_parser = subparsers.add_parser('add-topic', help='Automatically discover and add RSS feeds for a topic')
    add_topic_parser.add_argument('topic', help='Topic name for feed discovery')
    add_topic_parser.add_argument('--max-feeds', type=int, default=3, help='Maximum feeds to add')
    add_topic_parser.add_argument('--preview', action='store_true', help='Preview feeds before adding')
    add_topic_parser.add_argument('--dry-run', action='store_true', help='Show feeds without adding')
    add_topic_parser.add_argument('--region', choices=['us', 'uk', 'eu', 'apac', 'global'], default='global', help='Target region')
    
    # Discover feeds command (assistance)
    discover_feeds_parser = subparsers.add_parser('discover-feeds', help='Show how to find RSS feeds manually')
    
    # Search feeds command (discovery mode)
    search_feeds_parser = subparsers.add_parser('search-feeds', help='Search for RSS feeds for a topic (discovery mode)')
    search_feeds_parser.add_argument('topic', help='Topic to search RSS feed information for')

    # Topic status command
    topic_status_parser = subparsers.add_parser('topic-status', help='Show cache status for a topic')
    topic_status_parser.add_argument('topic', help='Topic to check status for')

    # Topic retry command
    topic_retry_parser = subparsers.add_parser('topic-retry', help='Force re-discovery of a topic (skip cache)')
    topic_retry_parser.add_argument('topic', help='Topic to re-discover')
    topic_retry_parser.add_argument('--max-feeds', type=int, default=5, help='Maximum feeds to discover')

    # Cache management command group
    cache_parser = subparsers.add_parser('cache', help='Manage feed discovery cache')
    cache_subparsers = cache_parser.add_subparsers(dest='cache_command', help='Cache operations')

    # Cache list command
    cache_list_parser = cache_subparsers.add_parser('list', help='List all cached topics')

    # Cache clear command
    cache_clear_parser = cache_subparsers.add_parser('clear', help='Clear all cached feeds')

    # Cache stale command
    cache_stale_parser = cache_subparsers.add_parser('stale', help='Show stale cache entries (>30 days)')

    # Cache refresh command
    cache_refresh_parser = cache_subparsers.add_parser('refresh', help='Re-discover stale topics')

    # Enhanced multi-keyword search command
    multi_parser = subparsers.add_parser('multi', help='Enhanced multi-keyword search with intersection scoring')
    multi_parser.add_argument('keywords', nargs='+', help='Keywords to search (e.g., ai insurance healthcare)')
    multi_parser.add_argument('--region', choices=['us', 'uk', 'eu', 'apac', 'global'], 
                             default='global', help='Filter by region for enhanced relevance')
    multi_parser.add_argument('--min-score', type=float, default=0.1, 
                             help='Minimum relevance score threshold (0.0-1.0)')
    multi_parser.add_argument('--limit', type=int, default=20, help='Maximum number of articles to show')
    multi_parser.add_argument('--details', action='store_true', 
                             help='Show detailed match information including keyword contexts')
    
    # Enhanced demo command
    demo_parser = subparsers.add_parser('demo', help='Demonstrate enhanced multi-keyword capabilities')
    demo_parser.add_argument('--region', choices=['us', 'uk', 'eu', 'apac', 'global'], 
                           help='Demo specific region (optional)')
    demo_parser.add_argument('--verbose', action='store_true', help='Show verbose demo output')
    
    # Parse arguments
    args = parser.parse_args()
    
    def _check_and_collect_fresh_data(database: Database, days: int) -> None:
        """
        Check if database has fresh articles, auto-collect if stale.
        
        Args:
            database: Database instance to check
            days: Number of days to consider as 'fresh'
        """
        try:
            articles = database.get_articles(limit=1)
            
            should_collect = False
            if not articles:
                should_collect = True
                reason = "Database is empty"
            else:
                newest_article = articles[0]
                if newest_article.published_at:
                    if newest_article.published_at.tzinfo:
                        article_date = newest_article.published_at.astimezone(None).replace(tzinfo=None)
                    else:
                        article_date = newest_article.published_at
                    article_age = datetime.now().replace(tzinfo=None) - article_date
                    age_days = article_age.days
                    if age_days > days:
                        should_collect = True
                        reason = f"Newest article is {age_days} days old (threshold: {days} days)"
                else:
                    should_collect = True
                    reason = "Articles have no timestamp"
            
            if should_collect:
                print(f"⚠ {reason}")
                print("📰 Collecting fresh articles...")
                
                try:
                    from .collector import SimpleCollector
                    from .config import Config
                    
                    config = Config()
                    collector = SimpleCollector(database)
                    total_stats = {"feeds_processed": 0, "total_fetched": 0, "total_added": 0, "ai_relevant_added": 0}
                    
                    for region_code, region_config in config.regions.items():
                        if region_config.enabled:
                            region_stats = collector.collect_region(config, region_code)
                            total_stats["feeds_processed"] += region_stats["feeds_processed"]
                            total_stats["total_fetched"] += region_stats["total_fetched"]
                            total_stats["total_added"] += region_stats["total_added"]
                            total_stats["ai_relevant_added"] += region_stats["ai_relevant_added"]
                    
                    print(f"✓ Collection complete: {total_stats['total_added']} articles from {total_stats['feeds_processed']} feeds")
                    print("Generating digest...")
                except Exception as e:
                    print(f"⚠ Collection failed: {e}")
                    print("Continuing with digest generation...")
                    
        except Exception as e:
            print(f"⚠ Error checking data freshness: {e}")
    
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

    # Auto-migrate database if needed
    from .migrations import get_database_migration_status, migrate_database
    try:
        migration_status = get_database_migration_status(str(db_path))
        if migration_status['needs_migration']:
            print(f"📦 Updating database schema (v{migration_status['current_version']} → v{migration_status['latest_version']})...")
            migrate_database(str(db_path), backup_before=False)
    except Exception as e:
        logger.warning(f"Database migration check failed: {e}")

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

        elif args.command == 'topics':
            # Topic management commands
            cmd_handle_topics(args, config, database)

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
                    start_date = datetime.now().replace(tzinfo=None) - timedelta(days=7)
                
                print(f"Generating weekly digest starting {start_date.strftime('%Y-%m-%d')}...")
                content = md_gen.generate_weekly_digest(start_date)
                
            elif args.type == 'topic':
                if not args.topic:
                    print("Error: Topic is required for topic analysis")
                    return
                
                # Handle multiple topics as list
                topics = args.topic if isinstance(args.topic, list) else [args.topic]
                topics_str = ', '.join(topics)
                
                print(f"Generating topic analysis for: {topics_str}")
                print(f"Time range: Last {args.days} days")

                # Smart auto-collection: collect fresh articles if data is stale
                _check_and_collect_fresh_data(database, args.days)

                # Check if spaCy mode is enabled (default: yes, unless --no-spacy)
                use_spacy = not getattr(args, 'no_spacy', False)
                
                if use_spacy:
                    print("Mode: spaCy semantic analysis")
                    
                    try:
                        # Import SpacyDigestAnalyzer
                        from .spacy_digest_analyzer import create_spacy_digest_analyzer
                        
                        # Initialize analyzer
                        analyzer = create_spacy_digest_analyzer(
                            cache_db_path=str(db_path),
                            ttl_hours=6
                        )
                        
                        if analyzer and analyzer._spacy_available:
                            # Get articles from database
                            all_articles = database.get_articles(limit=500)
                            
                            # Filter articles by date range
                            start_date = datetime.now().replace(tzinfo=None) - timedelta(days=args.days)
                            recent_articles = []
                            for a in all_articles:
                                if not a.published_at:
                                    recent_articles.append(a)
                                else:
                                    if a.published_at.tzinfo:
                                        article_date = a.published_at.astimezone(None).replace(tzinfo=None)
                                    else:
                                        article_date = a.published_at
                                    if article_date >= start_date:
                                        recent_articles.append(a)
                            
                            # Convert articles to dict format for analyzer
                            articles_dict = [
                                {
                                    'id': a.id,
                                    'title': a.title,
                                    'content': a.content or '',
                                    'summary': a.summary or '',
                                    'url': a.url,
                                    'source_name': a.source_name,
                                    'author': a.author,
                                    'published_at': a.published_at,
                                    'category': a.category,
                                    'ai_relevant': a.ai_relevant,
                                    'ai_keywords_found': a.ai_keywords_found or []
                                }
                                for a in recent_articles
                            ]
                            
                            # Analyze with spaCy
                            print("Analyzing articles with spaCy...")
                            scored_articles = analyzer.analyze(
                                articles=articles_dict,
                                topics=topics,
                                days=args.days,
                                use_and_logic=True
                            )
                            
                            # Check cache status
                            if hasattr(analyzer, 'cache'):
                                cache_key = analyzer.cache._generate_cache_key(topics, args.days)
                                cached = analyzer.cache.get(topics, args.days)
                                if cached:
                                    print("Cache: HIT (using cached results)")
                                else:
                                    print("Cache: MISS (fresh analysis)")
                            
                            # Generate spaCy-powered digest
                            if scored_articles:
                                print(f"Found {len(scored_articles)} relevant articles (confidence ≥ 70%)")
                                content = md_gen.generate_spacy_topic_digest(
                                    topics=topics,
                                    scored_articles=scored_articles,
                                    days=args.days
                                )
                            else:
                                # Fallback to keyword-based if no high-confidence articles
                                print("No high-confidence matches found, using keyword-based digest")
                                print("Fallback: Keyword matching")
                                content = _generate_keyword_topic_digest(md_gen, database, topics, args.days)
                        
                        else:
                            # spaCy not available, use keyword mode
                            print("spaCy not available, using keyword-based analysis")
                            print("Fallback: Keyword matching")
                            content = _generate_keyword_topic_digest(md_gen, database, topics, args.days)
                    
                    except Exception as e:
                        # Error with spaCy, fallback to keywords
                        print(f"spaCy analysis failed: {e}")
                        print("Fallback: Keyword matching")
                        content = _generate_keyword_topic_digest(md_gen, database, topics, args.days)
                
                else:
                    # Keyword-only mode (--no-spacy flag)
                    print("Mode: Keyword-only matching (--no-spacy)")
                    content = _generate_keyword_topic_digest(md_gen, database, topics, args.days)
            
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
            
        # Feed discovery commands
        elif args.command == 'add-topic':
            handle_add_topic_command(args, config, database)
            
        elif args.command == 'discover-feeds':
            handle_discover_feeds_command()
            
        elif args.command == 'search-feeds':
            handle_search_feeds_command(args)

        elif args.command == 'topic-status':
            handle_topic_status_command(args, database)

        elif args.command == 'topic-retry':
            handle_topic_retry_command(args, database)

        elif args.command == 'cache':
            handle_cache_command(args, database)

        # Enhanced multi-keyword commands (with lazy loading)
        elif args.command == 'multi':
            handle_multi_command(args, database)
            
        elif args.command == 'websearch':
            handle_websearch_command(args, database)
            
        elif args.command == 'demo':
            handle_demo_command(args, database)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error executing command: {e}")
        sys.exit(1)


def cmd_handle_topics(args, config: Config, database: Database):
    """Handle topic management commands."""
    try:
        if args.topic_command == 'list':
            # List all topics
            topics = config.list_topics()

            if not topics:
                print("📋 No topics configured yet.")
                print("   Use 'ai-news topics add <name> <keywords>' to add topics")
                return

            print(f"\n📋 Configured Topics ({len(topics)}):")
            print("=" * 60)

            for topic_name in sorted(topics):
                topic = config.topics[topic_name]

                if args.verbose:
                    print(f"\n{topic_name}:")
                    print(f"   Keywords: {', '.join(topic.keywords[:10])}")
                    if len(topic.keywords) > 10:
                        print(f"             ... and {len(topic.keywords) - 10} more")
                    print(f"   Auto-discover: {topic.auto_discover}")
                    print(f"   Min confidence: {topic.min_confidence}")
                else:
                    keyword_preview = ', '.join(topic.keywords[:5])
                    if len(topic.keywords) > 5:
                        keyword_preview += f" ... (+{len(topic.keywords) - 5})"
                    print(f"  • {topic_name}: {keyword_preview}")

        elif args.topic_command == 'add':
            # Add a new topic
            topic = config.add_topic(
                name=args.name,
                keywords=args.keywords,
                auto_discover=not args.no_discover
            )

            print(f"\n✅ Topic '{args.name}' added successfully!")
            print(f"   Keywords: {', '.join(topic.keywords)}")
            print(f"   Auto-discovery: {'enabled' if topic.auto_discover else 'disabled'}")

        elif args.topic_command == 'remove':
            # Remove a topic
            if config.remove_topic(args.name):
                print(f"\n✅ Topic '{args.name}' removed successfully!")
            else:
                print(f"\n❌ Topic '{args.name}' not found!")

        elif args.topic_command == 'discover':
            # Run topic discovery
            use_spacy = args.use_spacy and not args.no_spacy
            discovery = create_topic_discovery(database, use_spacy=use_spacy)

            if args.topic == 'all':
                # Discover for all topics with auto_discover enabled
                topics_to_discover = [
                    name for name, topic in config.topics.items()
                    if topic.auto_discover
                ]

                if not topics_to_discover:
                    print("📭 No topics with auto-discovery enabled.")
                    return

                print(f"\n🔍 Discovering for {len(topics_to_discover)} topics...")

                total_discovered = 0
                for topic_name in topics_to_discover:
                    topic = config.topics[topic_name]
                    articles = database.get_articles(limit=500)

                    count = discovery.learn_from_articles(
                        articles=articles,
                        topic_name=topic_name,
                        base_keywords=topic.keywords,
                        min_occurrence=args.min_occurrence
                    )

                    print(f"  • {topic_name}: {count} new discoveries")
                    total_discovered += count

                    # Prune if requested
                    if args.prune:
                        pruned = discovery.prune_stale_discoveries(topic_name)
                        if pruned > 0:
                            print(f"    Pruned {pruned} stale discoveries")

                print(f"\n✅ Discovery complete! Total new discoveries: {total_discovered}")

            else:
                # Discover for specific topic
                if args.topic not in config.topics:
                    print(f"❌ Topic '{args.topic}' not found!")
                    return

                topic = config.topics[args.topic]
                articles = database.get_articles(limit=500)

                print(f"\n🔍 Discovering keywords for topic: {args.topic}")
                print(f"   Base keywords: {', '.join(topic.keywords[:5])}")

                count = discovery.learn_from_articles(
                    articles=articles,
                    topic_name=args.topic,
                    base_keywords=topic.keywords,
                    min_occurrence=args.min_occurrence
                )

                print(f"\n✅ Discovered {count} new terms!")

                # Show statistics
                stats = discovery.get_discovery_stats(args.topic)
                print(f"   Total discoveries: {stats['total_discovered']}")
                print(f"   Average confidence: {stats['avg_confidence']}")

                # Prune if requested
                if args.prune:
                    pruned = discovery.prune_stale_discoveries(args.topic)
                    if pruned > 0:
                        print(f"   Pruned {pruned} stale discoveries")

        elif args.topic_command == 'stats':
            # Show discovery statistics
            discovery = create_topic_discovery(database)

            if args.topic == 'all':
                # Show stats for all topics
                print("\n📊 Discovery Statistics for All Topics")
                print("=" * 60)

                for topic_name in sorted(config.list_topics()):
                    stats = discovery.get_discovery_stats(topic_name)
                    print(f"\n{topic_name}:")
                    print(f"   Total discoveries: {stats['total_discovered']}")
                    print(f"   Avg confidence: {stats['avg_confidence']}")
                    if stats['last_updated']:
                        print(f"   Last updated: {stats['last_updated']}")
            else:
                # Show stats for specific topic
                if args.topic not in config.topics:
                    print(f"❌ Topic '{args.topic}' not found!")
                    return

                stats = discovery.get_discovery_stats(args.topic)

                print(f"\n📊 Discovery Statistics: {args.topic}")
                print("=" * 60)
                print(f"Total discoveries: {stats['total_discovered']}")
                print(f"Average confidence: {stats['avg_confidence']}")
                if stats['last_updated']:
                    print(f"Last updated: {stats['last_updated']}")

                # Get expanded keywords
                topic = config.topics[args.topic]
                expanded = discovery.get_expanded_keywords(
                    args.topic,
                    topic.keywords,
                    min_confidence=topic.min_confidence,
                    max_keywords=20
                )

                print(f"\n🔑 Expanded keywords (base + discovered):")
                for i, kw in enumerate(expanded[:20], 1):
                    discovered_marker = "✓" if kw not in topic.keywords else " "
                    print(f"  {i}. [{discovered_marker}] {kw}")

                if len(expanded) > 20:
                    print(f"  ... and {len(expanded) - 20} more")

        elif args.topic_command == 'suggest':
            # Suggest related topics
            if args.topic not in config.topics:
                print(f"❌ Topic '{args.topic}' not found!")
                return

            discovery = create_topic_discovery(database)
            topic = config.topics[args.topic]

            suggestions = discovery.suggest_related_topics(args.topic, topic.keywords)

            print(f"\n💡 Suggested Related Topics for '{args.topic}'")
            print("=" * 60)

            if suggestions:
                for suggestion in suggestions:
                    print(f"  • {suggestion}")
                print("\n💡 Use 'ai-news topics add' to create these combinations")
            else:
                print("  No strong suggestions found yet.")
                print("  Try running 'ai-news topics discover' first!")

        else:
            print("❌ Unknown topic command. Use --help to see available commands.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


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


# Web search command handler with intersection detection
def handle_websearch_command(args, database):
    """
    Handle web search command for single or multiple topics with intersection detection.

    Usage:
        ai-news websearch "healthcare"           # Single topic
        ai-news websearch "healthcare" "finance"  # Two topics + intersection
        ai-news websearch "ai" "robotics" "healthcare"  # Three topics + combos
    """
    try:
        topics = args.topics
        
        # Display search plan header
        print("\n" + "="*60)
        print("🔍 AI NEWS WEB SEARCH WITH INTERSECTION DETECTION")
        print("="*60)
        print(f"📋 Topics: {', '.join(topics)}")
        print(f"🎯 Max results per topic: {args.limit}")
        print(f"📊 Min confidence: {args.min_confidence}")
        
        # Import required modules
        from .search_collector import SearchEngineCollector
        from .intersection_optimization import create_intersection_optimizer
        from .intersection_planner import (
            plan_topic_searches,
            format_search_summary,
            estimate_total_searches
        )
        
        # Initialize components
        search_collector = SearchEngineCollector(database)
        optimizer = create_intersection_optimizer()
        
        # Plan the searches
        if args.no_intersections or len(topics) == 1:
            # Individual topics only - use list-based tags
            search_plans = [
                {'search_type': 'individual', 'topics': [t], 
                 'query': f'AI {t}', 'tags': ['AI', t]}
                for t in topics
            ]
            print(f"📌 Mode: Individual topics only")
        else:
            # Generate intersection combinations
            estimate = estimate_total_searches(len(topics), args.max_intersection_size)
            print(f"🔗 Mode: Intersection detection enabled")
            print(f"📊 Estimated searches: {estimate['total']}")
            print(f"   • Individual: {estimate['individual']}")
            print(f"   • Intersections: {estimate['intersections']}")
            
            search_plans = plan_topic_searches(
                topics, 
                max_intersection_size=args.max_intersection_size,
                min_intersection_size=2
            )
        
        print()
        print(format_search_summary(search_plans))
        print()
        
        # Execute searches and collect results
        all_results = []
        total_articles = 0
        rss_articles = []
        
        for i, plan in enumerate(search_plans, 1):
            result = _execute_search_plan(
                plan, i, len(search_plans),
                search_collector, optimizer, database,
                args.limit, args.min_confidence
            )
            all_results.append(result)
            total_articles += result['count']
        
        # Collect from RSS feeds if requested
        if args.include_rss:
            print("\n" + "="*60)
            print("📡 COLLECTING RSS FEEDS")
            print("="*60)
            
            from .collector import SimpleCollector
            from .config import Config
            
            # Parse regions
            regions = args.regions.split(',') if args.regions else ['global']
            regions = [r.strip() for r in regions]
            print(f"📌 Regions: {', '.join(regions)}")
            print()
            
            config = Config()
            rss_collector = SimpleCollector(database)
            
            for region in regions:
                print(f"📰 Collecting from {region.upper()} region...")
                try:
                    result = rss_collector.collect_region(config, region)
                    region_articles = result.get('articles', [])
                    rss_articles.extend(region_articles)
                    print(f"   ✅ Collected {len(region_articles)} articles")
                except Exception as e:
                    print(f"   ⚠️  Failed to collect from {region}: {e}")
            
            print(f"\n✅ Total RSS articles collected: {len(rss_articles)}")
        
        # Display summary
        _display_search_summary(all_results, total_articles, topics, rss_articles)
        
        # Save articles
        saved_count = _save_articles(all_results, database, args.save, rss_articles)
        
        if saved_count > 0:
            print(f"\n💡 You can now generate a digest with:")
            topics_str = ' '.join(topics[:2])  # Show first 2 topics
            print(f"   uv run python -m ai_news.cli digest --type topic --topic '{topics_str}' --days 1 --save")
        
    except KeyboardInterrupt:
        print("\n⚠️  Search interrupted by user")
    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
        print("💡 This could be due to:")
        print("   • No internet connection")
        print("   • Search engine limitations")
        print("   • Rate limiting")


def _execute_search_plan(
    plan: dict,
    index: int,
    total: int,
    search_collector,
    optimizer,
    database,
    limit: int,
    min_confidence: float
) -> dict:
    """Execute a single search plan and return results."""
    topics_str = ' + '.join(plan['topics'])
    tags = plan['tags']
    tags_display = ', '.join(tags)
    
    print(f"[{index}/{total}] 🔍 Searching: [{tags_display}]")
    print(f"   Query: {plan['query']}")
    
    # Perform search
    try:
        articles = search_collector.search_topic(
            ' '.join(plan['topics']),
            max_results=limit
        )
    except Exception as e:
        print(f"   ⚠️  Search failed: {e}")
        return {
            'plan': plan,
            'articles': [],
            'count': 0,
            'tags': tags,
            'error': str(e)
        }
    
    # Filter/validate intersections for multi-topic searches
    if plan['search_type'] == 'intersection' and articles:
        original_count = len(articles)
        articles = _filter_intersection_articles(
            articles, plan['topics'], optimizer, min_confidence
        )
        if original_count > 0:
            print(f"   🔬 Intersection validation: {len(articles)}/{original_count} articles passed")
    
    # Tag articles with the tag list (store in ai_keywords_found)
    for article in articles:
        if not article.ai_keywords_found:
            article.ai_keywords_found = []
        # Add tags to the article
        for tag in tags:
            if tag not in article.ai_keywords_found:
                article.ai_keywords_found.append(tag)
        # Set category to a readable format
        article.category = tags_display
    
    result = {
        'plan': plan,
        'articles': articles,
        'count': len(articles),
        'tags': tags
    }
    
    print(f"   ✅ Found {len(articles)} articles for [{tags_display}]")
    print()
    
    return result


def _filter_intersection_articles(
    articles,
    topics: list,
    optimizer,
    min_confidence: float
) -> list:
    """Filter articles that match intersection criteria."""
    filtered = []
    
    for article in articles:
        article_data = {
            'title': article.title,
            'content': article.content or '',
            'summary': article.summary
        }
        
        # Check intersection
        try:
            intersection_result = optimizer.detect_weighted_intersections(
                article_data, topics
            )
            
            if (intersection_result['intersection_detected'] and 
                intersection_result['confidence'] >= min_confidence):
                # Add confidence as metadata
                if not article.ai_keywords_found:
                    article.ai_keywords_found = []
                article.ai_keywords_found.append(
                    f"intersection_confidence:{intersection_result['confidence']:.2f}"
                )
                filtered.append(article)
        except Exception as e:
            # If intersection detection fails, include article anyway
            # (better to have false positives than miss good articles)
            filtered.append(article)
    
    return filtered


def _display_search_summary(all_results: list, total_articles: int, topics: list, rss_articles: list = None) -> None:
    """Display a summary of all search results."""
    print("\n" + "="*60)
    print("                    COLLECTION SUMMARY")
    print("="*60)
    
    individual_count = sum(
        r['count'] for r in all_results 
        if r['plan']['search_type'] == 'individual'
    )
    intersection_count = sum(
        r['count'] for r in all_results 
        if r['plan']['search_type'] == 'intersection'
    )
    
    print(f"Search plans executed: {len(all_results)}")
    print(f"Individual topic articles: {individual_count}")
    print(f"Intersection articles: {intersection_count}")
    print(f"Web search articles: {total_articles}")
    
    # RSS stats
    if rss_articles is not None:
        print(f"RSS feed articles: {len(rss_articles)}")
        total_articles += len(rss_articles)
    
    print(f"Total articles collected: {total_articles}")
    
    # Show breakdown by tag
    print("\n🏷️  Articles by tags:")
    for result in all_results:
        if result['count'] > 0:
            tags_str = ', '.join(result['tags'])
            print(f"  • [{tags_str}]: {result['count']} articles")
    
    print("="*60)


def _save_articles(all_results: list, database, auto_save: bool = False, rss_articles: list = None) -> int:
    """Save articles to database and return count saved."""
    websearch_count = sum(r['count'] for r in all_results)
    rss_count = len(rss_articles) if rss_articles else 0
    total_to_save = websearch_count + rss_count
    
    if total_to_save == 0:
        print("\n❌ No articles to save.")
        return 0
    
    # Prompt for save if not auto-save
    if not auto_save:
        save_option = input(f"\n💾 Save {total_to_save} articles to database? (y/n): ").strip().lower()
        if save_option != 'y':
            print("Articles not saved.")
            return 0
    
    # Save websearch articles
    saved_count = 0
    for result in all_results:
        tags_str = ', '.join(result['tags'])
        for article in result['articles']:
            try:
                if database.save_article(article):
                    saved_count += 1
            except Exception as e:
                print(f"   ⚠️  Failed to save '{article.title[:50]}...': {e}")
    
    # Save RSS articles
    if rss_articles:
        for article in rss_articles:
            try:
                if database.save_article(article):
                    saved_count += 1
            except Exception as e:
                print(f"   ⚠️  Failed to save RSS article '{article.title[:50]}...': {e}")
    
    print(f"\n✅ Saved {saved_count}/{total_to_save} articles to database")
    print(f"🏷️  Articles tagged with topic lists")
    
    return saved_count


def _handle_arbitrary_multi_command(args, database):
    """Handle arbitrary topic collection within multi command."""
    try:
        print(f"🎯 Collecting articles for arbitrary topics: {' + '.join(args.keywords)}")
        
        # Lazy import required components
        from .intersection_optimization import create_intersection_optimizer
        from .search_collector import SearchEngineCollector
        
        # Initialize components
        optimizer = create_intersection_optimizer()
        search_collector = SearchEngineCollector(database)
        
        print(f"🌍 Region: {args.region.upper()}")
        print(f"📊 Minimum score: {args.min_score}")
        print(f"🔢 Limit: {args.limit}")
        print()
        
        # Process topics into groups
        topic_groups = _process_arbitrary_topics(args.keywords)
        
        print(f"📊 Analyzing existing database articles...")
        articles = database.get_articles(
            limit=1000,  # Get more articles for better matching
            region=args.region if args.region != 'global' else None
        )
        print(f"✅ Retrieved {len(articles)} articles")
        
        if len(articles) == 0:
            print("❌ No articles found in database for this region")
            print("💡 Try running 'uv run ai-news collect' to populate the database first")
            print("💡 Or use --search-web to find current articles")
            return
        
        # Analyze articles for topic intersections
        print(f"🔍 Analyzing articles for topic intersections...")
        intersection_articles = []
        
        for article in articles:
            article_data = {
                "title": article.title,
                "content": article.content or "",
                "summary": article.summary
            }
            
            # Check each topic group
            for topic_group in topic_groups:
                intersection_result = optimizer.detect_weighted_intersections(
                    article_data, topic_group
                )
                
                if (intersection_result["intersection_detected"] and 
                    intersection_result["confidence"] >= args.min_score):
                    
                    # Validate relevance
                    validation = optimizer.validate_intersection_relevance(
                        intersection_result, article_data
                    )
                    
                    if validation["is_relevant"]:
                        intersection_articles.append({
                            "article": article,
                            "confidence": intersection_result["confidence"],
                            "relevance_score": validation["relevance_score"],
                            "topic_group": topic_group,
                            "matches": len(intersection_result.get("matches", []))
                        })
                        break  # Don't add the same article multiple times
        
        # Sort by confidence
        intersection_articles.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Display results
        if not intersection_articles:
            print("❌ No articles found matching your topic intersections")
            print("💡 Try:")
            print("   • Lowering --min-score (default: 0.1)")
            print("   • Using different topic combinations")
            print("   • Running 'uv run ai-news collect' to populate database")
            return
        
        print(f"\n🎯 Found {len(intersection_articles)} articles with topic intersections:")
        print("=" * 80)
        
        for i, result in enumerate(intersection_articles[:args.limit], 1):
            article = result["article"]
            topic_group = " + ".join(result["topic_group"])
            
            print(f"\n{i}. {article.title}")
            print(f"   🔗 Topic intersection: {topic_group}")
            print(f"   📊 Confidence: {result['confidence']:.3f} | Relevance: {result['relevance_score']:.3f}")
            print(f"   💥 Matches: {result['matches']}")
            print(f"   📰 Source: {article.source_name} | Region: {article.region.upper()}")
            
            # Truncated summary
            if article.summary:
                summary = article.summary[:150] + "..." if len(article.summary) > 150 else article.summary
                print(f"   📝 {summary}")
            
            print(f"   🔗 {article.url}")
        
        print(f"\n✅ Successfully found {len(intersection_articles)} intersection articles")
        print(f"   🔍 Analyzed {len(articles)} total articles")
        print(f"   🎯 Showing top {min(len(intersection_articles), args.limit)} results")
        
    except Exception as e:
        print(f"❌ Error during arbitrary topic collection: {e}")
        import traceback
        traceback.print_exc()


def _process_arbitrary_topics(topics: List[str]) -> List[List[str]]:
    """Process arbitrary topics into logical groups for intersection analysis."""
    topic_groups = []
    current_group = []
    
    for topic in topics:
        # If topic contains spaces, treat as single topic in its own group
        if ' ' in topic:
            if current_group:
                topic_groups.append(current_group)
                current_group = []
            topic_groups.append([topic])
        else:
            current_group.append(topic)
            # Group every 3 single-word topics together
            if len(current_group) >= 3:
                topic_groups.append(current_group)
                current_group = []
    
    # Add remaining topics
    if current_group:
        topic_groups.append(current_group)
    
    # If no groups were created, put all topics in one group
    if not topic_groups and topics:
        topic_groups.append([topics[0]] if len(topics) == 1 else topics[:2])
    
    return topic_groups


# Enhanced multi-keyword command handlers
def handle_multi_command(args, database):
    """Handle enhanced multi-keyword search command with arbitrary topic support."""
    try:
        print(f"🔍 Initializing enhanced multi-keyword search...")
        
        # Check if user provided arbitrary topics (not in predefined categories)
        predefined_categories = ['ai', 'ml', 'insurance', 'healthcare', 'fintech']
        user_topics = [k.lower() for k in args.keywords]
        
        # If user topics include non-predefined categories, use arbitrary mode
        if not all(topic in predefined_categories for topic in user_topics):
            print("🎯 Arbitrary topics detected - using arbitrary topic mode")
            return _handle_arbitrary_multi_command(args, database)
        
        # Original multi command logic for predefined categories
        # Lazy import enhanced collector
        from .enhanced_collector import EnhancedMultiKeywordCollector
        
        # Initialize enhanced collector
        enhanced_collector = EnhancedMultiKeywordCollector(performance_mode=True)
        print(f"✅ Enhanced collector initialized")
        
        # Build keyword categories from query parts
        categories = {}
        category_mapping = {
            'ai': enhanced_collector.categories['ai'].keywords,
            'ml': ['ML', 'machine learning', 'deep learning', 'neural network', 'algorithm'],
            'insurance': enhanced_collector.categories['insurance'].keywords,
            'healthcare': enhanced_collector.categories['healthcare'].keywords,
            'fintech': enhanced_collector.categories['fintech'].keywords
        }
        
        for keyword in args.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in category_mapping:
                categories[keyword_lower] = category_mapping[keyword_lower]
        
        if not categories:
            print("❌ No valid keyword categories found.")
            print("💡 Available categories: ai, ml, insurance, healthcare, fintech")
            print('💡 For arbitrary topics: ai-news multi "renewable energy" AI')
            return
        
        print(f"🔍 Enhanced multi-keyword search: {' + '.join(args.keywords)}")
        print(f"🌍 Region: {args.region.upper()}")
        print(f"📊 Minimum score: {args.min_score}")
        print()
        
        # Get articles from database
        print(f"📊 Fetching articles from database (region: {args.region.upper()})...")
        # Use appropriate limit based on args
        search_limit = min(1000, args.limit * 10)  # Get more articles for better matching
        articles = database.get_articles(limit=search_limit, region=args.region if args.region != 'global' else None)
        print(f"✅ Retrieved {len(articles)} articles")
        
        if len(articles) == 0:
            print("❌ No articles found in database for this region")
            print("💡 Try running 'uv run ai-news collect' to populate the database first")
            return
        
        # Filter articles using enhanced analysis
        print(f"🔍 Analyzing articles for relevance...")
        filtered_results = []
        
        for i, article in enumerate(articles):
            if i % 100 == 0 and i > 0:
                print(f"   Progress: {i}/{len(articles)} articles analyzed...")
                
            result = enhanced_collector.analyze_multi_keywords(
                title=article.title,
                content=article.content,
                categories=categories,
                region=args.region,
                min_score=args.min_score
            )
            
            if result.is_relevant:
                filtered_results.append((article, result))
        
        print(f"✅ Analysis complete: {len(filtered_results)} relevant articles found")
        
        # Sort by final score
        filtered_results.sort(key=lambda x: x[1].final_score, reverse=True)
        
        # Display results
        if not filtered_results:
            print("🔍 No articles found matching your criteria.")
            print("💡 Try lowering the minimum score with --min-score 0.05")
            return
        
        print(f"\n🎯 Found {len(filtered_results)} matching articles:")
        print("=" * 80)
        
        for i, (article, result) in enumerate(filtered_results[:args.limit], 1):
            # Article header
            relevance_indicator = "🤖" if article.ai_relevant else "  "
            print(f"{i}. {relevance_indicator} {article.title}")
            
            # Article metadata
            date_str = article.published_at.strftime("%Y-%m-%d") if article.published_at else "Unknown"
            print(f"   📅 {date_str} | 📰 {article.source_name} | 🌍 {article.region.upper()}")
            
            # Enhanced scores
            print(f"   📊 Final Score: {result.final_score:.3f}")
            print(f"   🎯 Total Score: {result.total_score:.3f} | Intersection: {result.intersection_score:.3f}")
            
            # Category scores
            if result.category_scores:
                categories_text = ", ".join([f"{cat}: {score:.2f}" for cat, score in result.category_scores.items()])
                print(f"   📈 Categories: {categories_text}")
            
            # Top keyword matches (if details requested)
            if args.details and result.matches:
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
        
        # Generate coverage report
        if filtered_results:
            print("\n" + "=" * 50)
            print("ENHANCED SEARCH SUMMARY")
            print("=" * 50)
            
            # Category statistics
            category_stats = {}
            for _, result in filtered_results[:args.limit]:
                for category, score in result.category_scores.items():
                    if category not in category_stats:
                        category_stats[category] = {'count': 0, 'total_score': 0}
                    category_stats[category]['count'] += 1
                    category_stats[category]['total_score'] += score
            
            for category, stats in category_stats.items():
                avg_score = stats['total_score'] / stats['count']
                print(f"{category.upper()}: {stats['count']} articles (avg score: {avg_score:.3f})")
            
            # Performance summary
            avg_score = sum(r.final_score for _, r in filtered_results[:args.limit]) / len(filtered_results[:args.limit])
            print(f"\nAverage relevance score: {avg_score:.3f}")
            print(f"High relevance articles (score > 0.5): {sum(1 for _, r in filtered_results[:args.limit] if r.final_score > 0.5)}")
            print("=" * 50)
        
    except ImportError as e:
        print(f"❌ Enhanced multi-keyword functionality not available: {e}")
        print("💡 Make sure enhanced_collector.py is available")
    except Exception as e:
        print(f"❌ Error during multi-keyword search: {e}")
        import traceback
        traceback.print_exc()


def handle_demo_command(args, database):
    """Handle enhanced demo command to showcase multi-keyword capabilities."""
    try:
        print("🎯 Initializing Enhanced Multi-Keyword Demo...")
        
        # Lazy import enhanced collector
        from .enhanced_collector import EnhancedMultiKeywordCollector
        print("✅ Enhanced collector imported")
        
        print("🎯 Enhanced Multi-Keyword Demo")
        print("=" * 60)
        print("Demonstrating advanced AI News search capabilities")
        print()
        enhanced_collector = EnhancedMultiKeywordCollector(performance_mode=True)
        print("✅ Enhanced collector initialized")
        
        # Simplified demo - just run one query
        print("\n🔍 Running simplified demo (AI + Insurance)...")
        print("-" * 40)
        
        # Build search categories
        search_categories = {
            'ai': enhanced_collector.categories['ai'].keywords
        }
        print("✅ Search categories built")
        
        # Get limited sample of articles (smaller for demo performance)
        demo_limit = 20  # Very small limit for quick demo
        print(f"📊 Fetching {demo_limit} articles...")
        articles = database.get_articles(limit=demo_limit, region='global')
        print(f"✅ Retrieved {len(articles)} articles")
        
        if not articles:
            print("  ❌ No articles found")
            return
        
        # Analyze articles
        matches = 0
        total_score = 0
        intersection_matches = 0
        high_relevance = 0
        
        print(f"🔍 Analyzing {len(articles)} articles...")
        for i, article in enumerate(articles):
            if i % 5 == 0:
                print(f"   Progress: {i}/{len(articles)}...")
                
            result = enhanced_collector.analyze_multi_keywords(
                title=article.title,
                content=article.content,
                categories=search_categories,
                region='global',
                min_score=0.05
            )
            
            if result.is_relevant:
                matches += 1
                total_score += result.final_score
                if result.intersection_score > 0:
                    intersection_matches += 1
                if result.final_score > 0.3:
                    high_relevance += 1
        
        coverage = (matches / len(articles)) * 100 if len(articles) > 0 else 0
        print(f"  📊 Articles analyzed: {len(articles)}")
        print(f"  🎯 Matches found: {matches} ({coverage:.1f}% coverage)")
        print(f"  🔗 Intersection matches: {intersection_matches}")
        print(f"  ⭐ High relevance (score > 0.3): {high_relevance}")
        
        if matches > 0:
            avg_score = total_score / matches
            print(f"  📈 Average relevance score: {avg_score:.3f}")
            print(f"  ✅ High quality articles: {(high_relevance/matches)*100:.1f}%")
        
        print("\n🎉 Demo completed successfully!")
        
        # Generate demo summary
        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        print(f"📊 Total queries analyzed: 1")
        print(f"🎯 Total matches found: {matches}")
        print(f"🌍 Regions covered: 1")
        print(f"🔍 Category combinations: 1")
        print()
        print("💡 Enhanced Features Demonstrated:")
        print("  • Multi-keyword intersection scoring")
        print("  • Regional relevance boosting")
        print("  • Category-specific keyword matching")
        print("  • Advanced relevance scoring")
        print("  • Performance-optimized analysis")
        print()
        print("🚀 Try it yourself:")
        print("  ai-news multi ai insurance --region uk")
        print("  ai-news multi ml healthcare --details")
        print("  ai-news multi ai fintech --min-score 0.2")
        print("=" * 60)
        
    except ImportError as e:
        print(f"❌ Enhanced demo functionality not available: {e}")
        print("💡 Make sure enhanced_collector.py is available")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


def handle_add_topic_command(args, config, database):
    """Handle automatic feed discovery and addition for a topic."""
    try:
        print(f"🔍 Searching for {args.topic} RSS feeds...")
        
        # Import our feed discovery engine
        from .feed_discovery import FeedDiscoveryEngine, NoFeedsFoundError
        
        discovery = FeedDiscoveryEngine(database)
        
        try:
            feeds = discovery.discover_feeds_for_topic(args.topic, args.max_feeds)
            
            if not feeds:
                print(f"❌ No RSS feeds found for '{args.topic}'")
                print("💡 Try a different topic or add feeds manually:")
                print(f"   uv run ai-news feeds add --name '{args.topic} Feed' --url 'RSS_URL'")
                return
            
            print(f"\n📡 Found {len(feeds)} RSS feed(s) for '{args.topic}':")
            
            # Show discovered feeds
            for i, feed in enumerate(feeds, 1):
                print(f"{i}. {feed['title']}")
                print(f"   🔗 {feed['url']}")
                print(f"   📊 Relevance: {feed['relevance_score']:.0%}")
                print(f"   📰 {feed['article_count']} articles")
                print()
            
            if args.preview or args.dry_run:
                print("📰 Preview recent articles:")
                for i, feed in enumerate(feeds[:2], 1):  # Show max 2 feeds
                    print(f"\n{feed['title']} (showing 3 recent articles):")
                    try:
                        from .feed_discovery import FeedValidator
                        validator = FeedValidator()
                        articles = validator.get_feed_preview(feed['url'], 3)
                        for j, article in enumerate(articles, 1):
                            print(f"   {j}. {article['title']}")
                    except Exception:
                        print(f"   ⚠️  Could not fetch articles")
                print()
            
            if args.dry_run:
                print("🔍 Dry run complete - no feeds were added")
                return
            
            # Add feeds to configuration
            added_count = 0
            
            for feed in feeds:
                try:
                    # Add to specified region
                    feed_name = f"{args.topic.title()} - {feed['title']}"
                    success = config.add_feed(
                        region=args.region,
                        name=feed_name,
                        url=feed['url'],
                        category=args.topic.lower(),
                        ai_keywords=args.topic.split() + ['AI', 'artificial intelligence']
                    )
                    
                    if success:
                        added_count += 1
                        print(f"✅ Added: {feed_name}")
                    else:
                        print(f"⚠️  Feed already exists: {feed_name}")
                        
                except Exception as e:
                    print(f"❌ Failed to add {feed['title']}: {e}")
            
            print(f"\n🎉 Successfully added {added_count}/{len(feeds)} feeds for '{args.topic}'")
            
            # Collect articles from new feeds
            if added_count > 0:
                print("📥 Collecting articles from new feeds...")
                from .collector import SimpleCollector
                collector = SimpleCollector(database)
                stats = collector.collect_region(config, args.region)
                
                # Test if topic works now
                print(f"🧪 Testing search for '{args.topic}':")
                articles = database.search_articles(args.topic, limit=3)
                if articles:
                    for i, article in enumerate(articles, 1):
                        print(f"{i}. {article.title}")
                else:
                    print("   No articles found yet - try again after the next collection cycle")
                
                print(f"\n📊 Collection Summary:")
                print(f"   Feeds processed: {stats['feeds_processed']}")
                print(f"   Articles added: {stats['total_added']}")
                print(f"   AI-relevant added: {stats['ai_relevant_added']}")
        
        except NoFeedsFoundError as e:
            print(f"❌ {e}")
            print("\n💡 Tips for finding RSS feeds:")
            
            # Check if we have related topics
            topic_lower = args.topic.lower()
            related_topics = []
            
            topic_suggestions = {
                'llm': ['artificial intelligence', 'technology', 'machine learning'],
                'large language model': ['artificial intelligence', 'technology', 'machine learning'],
                'insurance': ['fintech', 'finance', 'technology'],
                'healthcare it': ['healthcare', 'technology', 'fintech'],
                'crypto': ['blockchain', 'fintech', 'technology'],
                'sustainability': ['renewable energy', 'technology', 'environment'],
                'cybersecurity': ['technology', 'security', 'fintech']
            }
            
            for keyword, suggestions in topic_suggestions.items():
                if keyword in topic_lower:
                    related_topics.extend(suggestions)
                    break
            
            if related_topics:
                print("🎯 Try these related topics that have known feeds:")
                for topic in related_topics[:3]:
                    print(f"   • uv run ai-news add-topic '{topic}' --dry-run")
                print()
            
            print("🔍 Manual feed discovery guide:")
            print(f"1. Google: '{args.topic} RSS feed' or '{args.topic} blog feed'")
            print("2. Look for RSS icons (🟠) on industry websites")
            print("3. Check industry publications and blogs")
            print()
            print("✅ Add manually once found:")
            print(f"   uv run ai-news feeds add --name '{args.topic.title()} Blog' --url 'RSS_URL'")
            print()
            print("🛠️  Get help with discovery:")
            print("   uv run ai-news discover-feeds")  # For manual guidance
            print(f"   uv run ai-news search-feeds '{args.topic}'  # Search feed info")
    
    except Exception as e:
        print(f"❌ Error discovering feeds: {e}")
        print("💡 You can still add feeds manually:")
        print(f"   uv run ai-news feeds add --name '{args.topic} Feed' --url 'RSS_URL'")


def handle_discover_feeds_command():
    """Show assistance for manually finding RSS feeds."""
    print("🔍 How to find RSS feeds manually:")
    print()
    print("1. Search Google: '[your topic] RSS feed'")
    print("   Example: 'quantum computing RSS feed'")
    print()
    print("2. Look for RSS links on websites:")
    print("   🟠 RSS icon in browser")
    print("   🔗 Links ending in /rss, /feed, or .xml")
    print()
    print("3. Popular RSS directories:")
    print("   • Feedspot: https://blog.feedspot.com/")
    print("   • Feedly: https://feedly.com/")
    print("   • Inoreader: https://www.inoreader.com/")
    print()
    print("4. Example manual addition:")
    print("   uv run ai-news feeds add ")
    print("     --name 'Quantum Computing News' ")
    print("     --url 'https://example.com/quantum-rss.xml' ")
    print("     --category quantum --ai-keywords 'quantum,AI'")
    print()
    print("💡 Or use automatic discovery:")
    print("   uv run ai-news add-topic 'your-topic'")


def handle_search_feeds_command(args):
    """Search for RSS feed information for a topic (discovery mode)."""
    print(f"🔍 Searching for {args.topic} RSS feed information...")
    print("💡 This shows where to find feeds - not automatic addition")
    print()
    
    # Use existing websearch to find RSS feed information  
    from .search_collector import SearchEngineCollector
    from .database import Database
    
    # Create a temporary database instance for search
    db_path = 'data/production/ai_news.db'
    temp_db = Database(db_path)
    searcher = SearchEngineCollector(temp_db)
    
    search_queries = [
        f"{args.topic} RSS feed",
        f"best {args.topic} news sources RSS",
        f"{args.topic} news aggregator feed"
    ]
    
    for i, query in enumerate(search_queries, 1):
        try:
            print(f"📋 Results {i}: {query}")
            # Use SearXNG for better search results
            search_results = searcher.search_searxng(query, max_results=3)
            
            if search_results:
                for result in search_results:
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No description')
                    url = result.get('url', 'No URL')
                    engine = result.get('engine', ['unknown'])
                    print(f"   🔗 {title}")
                    print(f"   📍 {url}")
                    print(f"   💡 {content[:150]}...")
                    print(f"   🔍 Source: {', '.join(engine)}")
                    print()
            else:
                print("   No results found")
                print()
            
            if i < len(search_queries):
                print("-" * 50)
        
        except Exception as e:
            print(f"   ⚠️  Search error: {e}")
            print()
    
    # Auto-discover feeds from promising results
    print("\n🔍 Auto-discovering RSS feeds from promising results...")
    from .feed_discovery import FeedDiscoveryEngine
    from .database import Database
    
    try:
        db = Database('data/production/ai_news.db')
        discovery = FeedDiscoveryEngine(db)
        
        all_feeds = set()
        # Re-run search to get URLs for discovery
        for query in search_queries[:1]:  # Just check first query
            search_results = searcher.search_searxng(query, max_results=5)
            for result in search_results:
                url = result.get('url', '')
                title = result.get('title', '')
                content = result.get('content', '')
                
                if discovery._is_promising_feed_directory(url, title, content):
                    print(f"📂 Exploring: {title} ({url})")
                    discovered = discovery._discover_feeds_from_page(url)
                    all_feeds.update(discovered[:3])  # Just show first 3 per source
    
        if all_feeds:
            print(f"\n✅ Found {len(all_feeds)} RSS feeds:")
            for i, feed in enumerate(list(all_feeds)[:5], 1):  # Show first 5
                print(f"   {i}. {feed}")
                
            print(f"\n💡 Add them like this:")
            feeds_list = list(all_feeds)
            print(f"   uv run ai-news feeds add --name '{args.topic.title()} Feed 1' --url '{feeds_list[0]}'")
            if len(feeds_list) > 1:
                print(f"   uv run ai-news feeds add --name '{args.topic.title()} Feed 2' --url '{feeds_list[1]}'")
        else:
            print("\n😔 No RSS feeds found in the search results")
            
    except Exception as e:
        print(f"\n❌ Auto-discovery failed: {e}")
    
    print("\n💡 Manual addition:")
    print(f"   uv run ai-news feeds add --name '{args.topic} News' --url 'FEED_URL'")
    print()
    print("🚀 Or try automatic discovery:")
    print(f"   uv run ai-news add-topic '{args.topic}'")


def handle_topic_status_command(args, database):
    """Show cache status for a topic."""
    from .feed_discovery import FeedDiscoveryEngine

    print(f"\n📊 Checking cache status for: {args.topic}")

    engine = FeedDiscoveryEngine(database=database)

    if engine.cache.is_cache_fresh(args.topic):
        print(f"✅ Topic '{args.topic}' is cached and fresh")

        feeds = engine.cache.check_cache(args.topic)
        if feeds:
            print(f"\n📰 Cached feeds: {len(feeds)}")

            for feed in feeds:
                emoji = "🟢" if feed['relevance_score'] >= 0.7 else "🟡" if feed['relevance_score'] >= 0.4 else "🟠"
                print(f"{emoji} {feed['title']} ({feed['article_count']} articles)")
    else:
        print(f"❌ Topic '{args.topic}' is not cached or cache is stale")
        print(f"💡 Run 'ai-news topic-retry \"{args.topic}\"' to discover feeds")


def handle_topic_retry_command(args, database):
    """Force re-discovery of a topic (skip cache)."""
    from .feed_discovery import FeedDiscoveryEngine, NoFeedsFoundError

    print(f"\n🔄 Re-discovering feeds for '{args.topic}'...")

    engine = FeedDiscoveryEngine(database=database)

    try:
        feeds = engine.discover_feeds_for_topic(args.topic, max_feeds=args.max_feeds, force_discovery=True)

        print(f"\n✅ Found {len(feeds)} feeds for '{args.topic}'\n")

        for feed in feeds:
            emoji = "🟢" if feed['relevance_score'] >= 0.7 else "🟡" if feed['relevance_score'] >= 0.4 else "🟠"
            print(f"{emoji} {feed['title']}")
            print(f"   {feed['article_count']} articles • {feed['url']}")

        print(f"\n💾 Cache updated")

    except NoFeedsFoundError:
        print(f"\n❌ No feeds found for '{args.topic}'")
        print(f"\n💡 Suggestions:")
        print(f"   → Try a broader topic")
        print(f"   → Try related terms")
        print(f"   → Check spelling")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def handle_cache_command(args, database):
    """Handle cache management commands."""
    import sqlite3
    from .feed_discovery import FeedDiscoveryEngine

    engine = FeedDiscoveryEngine(database=database)

    if args.cache_command == 'list':
        topics = engine.cache.get_all_cached_topics()

        if not topics:
            print("No cached topics")
            return

        print(f"\n💾 Cached topics ({len(topics)}):\n")

        for topic in topics:
            is_fresh = engine.cache.is_cache_fresh(topic)
            status = "✅ Fresh" if is_fresh else "⚠️  Stale"
            print(f"{status}: {topic}")

    elif args.cache_command == 'clear':
        print("\n⚠️  This will clear all cached feeds.")
        response = input("Are you sure? (yes/no): ").lower().strip()

        if response not in ['yes', 'y']:
            print("Cancelled")
            return

        # Delete all from discovered_feeds
        with sqlite3.connect(database.db_path) as conn:
            conn.execute("DELETE FROM discovered_feeds")

        print("🧹 Cache cleared")

    elif args.cache_command == 'stale':
        # Get stale entries
        with sqlite3.connect(database.db_path) as conn:
            stale = conn.execute("""
                SELECT DISTINCT topic FROM discovered_feeds
                WHERE last_seen < date('now', '-30 days')
            """).fetchall()

        if not stale:
            print("✅ No stale entries")
            return

        print(f"\n⚠️  Stale entries ({len(stale)}):\n")

        for row in stale:
            print(f"   {row[0]}")

        print(f"\n💡 Run 'ai-news cache refresh' to update")

    elif args.cache_command == 'refresh':
        # Get stale entries
        with sqlite3.connect(database.db_path) as conn:
            stale = conn.execute("""
                SELECT DISTINCT topic FROM discovered_feeds
                WHERE last_seen < date('now', '-30 days')
            """).fetchall()

        if not stale:
            print("✅ No stale entries to refresh")
            return

        print(f"\n🔄 Refreshing {len(stale)} stale topics...\n")

        for row in stale:
            topic = row[0]
            print(f"Refreshing: {topic}")

            try:
                feeds = engine.discover_feeds_for_topic(topic, force_discovery=True)
                print(f"✅ Found {len(feeds)} feeds\n")
            except Exception as e:
                print(f"❌ Failed: {e}\n")

        print("✅ Refresh complete")

    else:
        print("❌ Unknown cache command. Use --help to see available commands.")


def _generate_keyword_topic_digest(md_gen: MarkdownGenerator, database: Database, topics: list, days: int, use_and_logic: bool = True) -> str:
    """
    Generate a keyword-based topic digest (fallback when spaCy unavailable or disabled).

    Args:
        md_gen: MarkdownGenerator instance
        database: Database instance
        topics: List of topic keywords
        days: Number of days for analysis
        use_and_logic: If True, articles must match ALL topics (AND). If False, match ANY topic (OR).

    Returns:
        Markdown digest content
    """
    from datetime import timedelta

    start_date = datetime.now().replace(tzinfo=None) - timedelta(days=days)
    topics_str = ', '.join(topics)

    if use_and_logic and len(topics) > 1:
        # AND logic: Articles must match ALL topics
        # Search for first topic to get candidate articles
        candidate_articles = database.search_articles(topics[0], limit=500)

        # Filter candidates that contain ALL remaining topics
        matching_articles = []
        for article in candidate_articles:
            article_text = f"{article.title} {article.content} {article.summary}".lower()
            # Check if ALL topics are found in this article
            if all(topic.lower() in article_text for topic in topics[1:]):
                matching_articles.append(article)

        unique_articles = matching_articles
    else:
        # OR logic: Search for articles related to any topic
        all_articles = []
        for topic in topics:
            articles = database.search_articles(topic, limit=100)
            all_articles.extend(articles)

        # Remove duplicates (by URL)
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)

    # Filter by date range
    recent_articles = []
    for a in unique_articles:
        if not a.published_at:
            recent_articles.append(a)
        else:
            if a.published_at.tzinfo:
                article_date = a.published_at.astimezone(None).replace(tzinfo=None)
            else:
                article_date = a.published_at
            if article_date >= start_date:
                recent_articles.append(a)

    if not recent_articles:
        return f"""# Topic Analysis: {topics_str}
*Last {days} days* - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}

*No articles found for '{topics_str}' in the last {days} days.*
"""

    # Sort by date (newest first), handling timezone-aware datetimes
    def sort_key(article):
        if article.published_at:
            if article.published_at.tzinfo:
                return article.published_at.astimezone(None).replace(tzinfo=None)
            return article.published_at
        return datetime.min

    recent_articles.sort(key=sort_key, reverse=True)

    # Generate digest
    logic_mode = "AND (all topics must match)" if use_and_logic and len(topics) > 1 else "OR (any topic matches)"
    digest = f"""# Topic Analysis: {topics_str}
*Last {days} days* - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Method:** Keyword-based matching ({logic_mode})

## 📈 Overview

- **Total Articles:** {len(recent_articles)}
- **Topics:** {topics_str}
- **Coverage Period:** {start_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}

## 📰 Articles ({len(recent_articles)})

"""

    for i, article in enumerate(recent_articles[:50], 1):  # Limit to 50 articles
        ai_indicator = "🤖 " if article.ai_relevant else ""
        date_str = article.published_at.strftime('%Y-%m-%d') if article.published_at else 'Unknown'

        digest += f"""### {i}. {ai_indicator}{article.title}

**Source:** {article.source_name} | **Date:** {date_str} | **Category:** {article.category}

{md_gen.generate_article_summary(article)}

**Read more:** [{article.url}]({article.url})

"""
        if article.ai_keywords_found:
            digest += f"**AI Keywords:** {', '.join(article.ai_keywords_found)}\n\n"

    return digest


if __name__ == '__main__':
    main()