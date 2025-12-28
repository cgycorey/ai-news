#!/usr/bin/env python3
"""Quick profiling of RSS collection performance."""

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

    # Get first 3 feeds
    global_region = config.regions.get("global")
    if not global_region or not global_region.feeds:
        print("No feeds found")
        return

    feeds_to_test = [f for f in global_region.feeds if f.enabled][:3]

    print("="*70)
    print("RSS COLLECTION PROFILING REPORT")
    print("="*70)

    total_feed_time = 0
    total_articles = 0
    network_times = []
    parse_times = []
    article_times = []

    for feed in feeds_to_test:
        print(f"\n📡 Feed: {feed.name}")
        print(f"   URL: {feed.url}")

        # Time network fetch
        start = time.time()
        root = collector.fetch_rss_feed(feed.url)
        fetch_time = time.time() - start
        network_times.append(fetch_time)

        print(f"   Network fetch: {fetch_time:.2f}s", flush=True)

        if not root:
            print("   ⚠️  Failed to fetch feed")
            continue

        # Time parsing
        start = time.time()
        articles = collector.fetch_feed(feed, max_articles=10)
        parse_time = time.time() - start
        parse_times.append(parse_time)

        print(f"   Parse time: {parse_time:.2f}s")
        print(f"   Articles fetched: {len(articles)}")

        # Time first article save
        if articles:
            article = articles[0]

            # Confidence scoring
            start = time.time()
            confidence = collector.confidence_scorer.calculate_confidence(article)
            conf_time = time.time() - start

            # Database save
            start = time.time()
            article_id = database.save_article(article)
            db_time = time.time() - start

            total_article_time = conf_time + db_time
            article_times.append(total_article_time)

            print(f"\n   First Article Breakdown:")
            print(f"     Confidence scoring: {conf_time:.3f}s")
            print(f"     Database save: {db_time:.3f}s")
            print(f"     Total: {total_article_time:.3f}s")

            total_articles += 1

        feed_total = fetch_time + parse_time
        total_feed_time += feed_total
        print(f"   Feed total: {feed_total:.2f}s", flush=True)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if network_times:
        avg_network = sum(network_times) / len(network_times)
        avg_parse = sum(parse_times) / len(parse_times)
        avg_article = sum(article_times) / len(article_times) if article_times else 0

        print(f"Feeds tested: {len(feeds_to_test)}")
        print(f"Total time: {total_feed_time:.2f}s")
        print(f"\nAverage times:")
        print(f"  Network fetch: {avg_network:.2f}s")
        print(f"  Parsing: {avg_parse:.2f}s")
        print(f"  Article save: {avg_article:.3f}s")

        # Identify bottlenecks
        print(f"\nBOTTLENECK ANALYSIS:")
        if avg_network > 5:
            print(f"  🔴 CRITICAL: Network fetches averaging {avg_network:.1f}s")
            print(f"     → Slow feeds or high timeouts (30s)")
            print(f"     → Recommendation: Disable slow feeds or reduce timeout")

        if avg_article > 0.5:
            print(f"  🟡 HIGH: Article saves averaging {avg_article:.2f}s")
            print(f"     → Confidence scoring or entity extraction slow")
            print(f"     → Recommendation: Disable auto-tagging during collection")

        if avg_network < 2 and avg_article < 0.2:
            print(f"  ✅ Performance looks good")

if __name__ == "__main__":
    main()
