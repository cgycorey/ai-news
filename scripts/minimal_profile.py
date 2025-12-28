#!/usr/bin/env python3
"""Minimal profiling of RSS collection to identify bottlenecks."""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_news.config import Config
from ai_news.collector import SimpleCollector
from ai_news.database import Database

def main():
    config = Config.load(Path("config.json"))
    database = Database(config.database_path)
    collector = SimpleCollector(database)

    # Get first feed
    global_region = config.regions.get("global")
    if not global_region or not global_region.feeds:
        print("No feeds found")
        return

    first_feed = global_region.feeds[0]

    print("="*70)
    print("MINIMAL RSS COLLECTION PROFILING")
    print("="*70)
    print(f"\nFeed: {first_feed.name}")
    print(f"URL: {first_feed.url}\n")

    # Step 1: Network fetch
    print("Step 1: Network fetch", flush=True)
    start = time.time()
    root = collector.fetch_rss_feed(first_feed.url)
    fetch_time = time.time() - start
    print(f"  Time: {fetch_time:.3f}s")

    if not root:
        print("  FAILED")
        return

    # Step 2: Parse feed (this is where fetch_feed processes articles)
    print("\nStep 2: Parse feed and process articles", flush=True)
    start = time.time()
    articles = collector.fetch_feed(first_feed, max_articles=10)
    parse_time = time.time() - start
    print(f"  Time: {parse_time:.3f}s")
    print(f"  Articles fetched: {len(articles)}")

    # Step 3: Save articles (with timing breakdown)
    if articles:
        print(f"\nStep 3: Save articles to database", flush=True)

        for i, article in enumerate(articles[:3], 1):
            print(f"\n  Article {i}: {article.title[:50]}...")

            # 3a. Confidence scoring
            start = time.time()
            confidence = collector.confidence_scorer.calculate_confidence(article)
            conf_time = time.time() - start
            print(f"    Confidence scoring: {conf_time:.3f}s (score: {confidence:.2f})")

            # 3b. Database save
            start = time.time()
            article_id = database.save_article(article)
            db_time = time.time() - start
            print(f"    Database save: {db_time:.3f}s")

            if i == 1:
                print(f"    (Note: First article includes auto-tagging)")

    # Summary
    total_time = fetch_time + parse_time
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Network fetch: {fetch_time:.3f}s ({fetch_time/total_time*100:.1f}%)")
    print(f"Parse & process: {parse_time:.3f}s ({parse_time/total_time*100:.1f}%)")
    print(f"Total: {total_time:.3f}s")

    # Estimate for 24 feeds
    estimated_total = total_time * 24
    print(f"\nEstimated time for 24 feeds: {estimated_total:.1f}s ({estimated_total/60:.1f} minutes)")

    # Bottleneck analysis
    print(f"\nBOTTLENECK ANALYSIS:")
    if fetch_time > 5:
        print(f"  🔴 CRITICAL: Network fetch slow ({fetch_time:.1f}s)")
    elif parse_time > 5:
        print(f"  🟡 HIGH: Article processing slow ({parse_time:.1f}s)")
    else:
        print(f"  ✅ Performance looks good")

if __name__ == "__main__":
    main()
