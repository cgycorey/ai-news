#!/usr/bin/env python3
"""Demo script for semantic filtering during collection."""

from src.ai_news.config import Config, FeedConfig
from src.ai_news.database import Database
from src.ai_news.collector import SimpleCollector

def demo_semantic_filtering():
    """Demonstrate semantic topic filtering during collection."""
    import tempfile
    import os

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        # Initialize database and collector
        db = Database(db_path)
        collector = SimpleCollector(db)

        # Create test feed configuration
        test_feed = FeedConfig(
            name="TechCrunch",
            url="https://techcrunch.com/feed/",
            category="tech",
            enabled=True
        )

        print("=" * 70)
        print("Semantic Filtering Demo")
        print("=" * 70)

        # Test 1: Collection without semantic filtering (default behavior)
        print("\n1. Collection WITHOUT semantic filtering (default):")
        print("-" * 70)
        stats = collector.collect_all_feeds([test_feed], max_articles_per_feed=5)
        print(f"   Articles fetched: {stats['total_fetched']}")
        print(f"   Articles added: {stats['total_added']}")

        # Test 2: Collection with semantic filtering for AI topics
        print("\n2. Collection WITH semantic filtering for 'AI', 'machine learning':")
        print("-" * 70)
        stats = collector.collect_all_feeds(
            [test_feed],
            max_articles_per_feed=5,
            topics=["artificial intelligence", "machine learning", "AI technology"]
        )
        print(f"   Articles fetched: {stats['total_fetched']}")
        print(f"   Articles added: {stats['total_added']}")
        print(f"   Semantically filtered: {stats.get('semantic_filtered', 0)}")

        print("\n" + "=" * 70)
        print("Demo complete!")
        print("=" * 70)
        print("\nNote: Semantic filtering is now available during collection.")
        print("      - Add 'topics=[...]'' parameter to collect_region(),")
        print("        collect_all_feeds(), or collect_multiple_regions()")
        print("      - Uses FastEmbed for semantic similarity (threshold >= 0.55)")
        print("      - Gracefully falls back to no filtering if FastEmbed not installed")
        print("      - Default behavior (no topics) remains unchanged")

    finally:
        # Cleanup
        try:
            os.unlink(db_path)
        except:
            pass

if __name__ == "__main__":
    demo_semantic_filtering()
