#!/usr/bin/env python3
"""Profile RSS collection performance to identify bottlenecks.

This script times each operation in the collection pipeline:
- Feed fetching (network I/O)
- Article processing (parsing, confidence scoring)
- Database saves (including entity extraction)
- Per-article operations breakdown

Usage:
    python scripts/profile_collection.py [--num-feeds N] [--output-report]
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import cProfile
import pstats
from io import StringIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_news.config import Config
from ai_news.collector import SimpleCollector
from ai_news.database import Database
from ai_news.article_tagger import get_article_tagger
from ai_news.entity_extractor import EntityExtractor
from ai_news.text_processor import TextProcessor


class CollectionProfiler:
    """Profile RSS collection performance."""

    def __init__(self, config_path: str = "config.json"):
        self.config = Config.load(Path(config_path))
        self.database = Database(self.config.database_path)
        self.collector = SimpleCollector(self.database)

        # Timing data
        self.timings = {
            "feed_fetch": [],
            "article_parsing": [],
            "confidence_scoring": [],
            "database_save": [],
            "entity_extraction": [],
            "auto_tagging": [],
            "per_article_total": [],
            "per_feed_total": []
        }

    def profile_feed_fetch(self, feed_config, max_articles: int = 50) -> Dict[str, Any]:
        """Profile a single feed fetch with detailed timing."""
        feed_times = {
            "feed_name": feed_config.name,
            "network_time": 0,
            "parse_time": 0,
            "article_count": 0,
            "article_times": [],
            "total_time": 0
        }

        start_total = time.time()

        # Time the network fetch
        start_network = time.time()
        root = self.collector.fetch_rss_feed(feed_config.url)
        feed_times["network_time"] = time.time() - start_network

        if root is None:
            feed_times["total_time"] = time.time() - start_total
            return feed_times

        # Time parsing
        start_parse = time.time()
        articles = self.collector.fetch_feed(feed_config, max_articles=max_articles)
        feed_times["parse_time"] = time.time() - start_parse
        feed_times["article_count"] = len(articles)

        # Profile per-article operations
        for i, article in enumerate(articles[:10]):  # Sample first 10 articles
            article_time = self.profile_article_save(article)
            feed_times["article_times"].append(article_time)

        feed_times["total_time"] = time.time() - start_total
        return feed_times

    def profile_article_save(self, article) -> Dict[str, Any]:
        """Profile saving a single article with breakdown."""
        times = {
            "confidence_scoring": 0,
            "database_transaction": 0,
            "entity_extraction": 0,
            "auto_tagging": 0,
            "total": 0
        }

        start_total = time.time()

        # Time confidence scoring
        start = time.time()
        confidence = self.collector.confidence_scorer.calculate_confidence(article)
        times["confidence_scoring"] = time.time() - start

        # Time database save (includes auto-tagging)
        start_db = time.time()

        # Temporarily disable auto-tagging to measure separately
        article_id = self.database.save_article(article, auto_tag=False)

        times["database_transaction"] = time.time() - start_db

        # Time entity extraction separately
        if article_id:
            start = time.time()
            tagger = get_article_tagger()
            tags = tagger.tag_article(article)
            times["entity_extraction"] = time.time() - start

            # Time tag saving
            start = time.time()
            tagger.save_tags(article_id, tags, self.database)
            times["auto_tagging"] = time.time() - start

        times["total"] = time.time() - start_total
        return times

    def profile_collection(self, num_feeds: int = 5) -> Dict[str, Any]:
        """Profile collection for first N feeds."""
        print(f"\n{'='*70}")
        print(f"PROFILING RSS COLLECTION (First {num_feeds} feeds)")
        print(f"{'='*70}\n")

        # Get global region feeds
        global_region = self.config.regions.get("global")
        if not global_region:
            print("Error: No global region found")
            return {}

        enabled_feeds = [f for f in global_region.feeds if f.enabled][:num_feeds]

        results = {
            "feeds": [],
            "summary": {},
            "bottlenecks": []
        }

        for feed in enabled_feeds:
            print(f"\n📡 Profiling feed: {feed.name}")
            print(f"   URL: {feed.url}")

            feed_result = self.profile_feed_fetch(feed, max_articles=50)
            results["feeds"].append(feed_result)

            # Print immediate results
            print(f"   Network fetch: {feed_result['network_time']:.2f}s")
            print(f"   Parse time: {feed_result['parse_time']:.2f}s")
            print(f"   Articles fetched: {feed_result['article_count']}")

            if feed_result["article_times"]:
                avg_article_time = sum(a["total"] for a in feed_result["article_times"]) / len(feed_result["article_times"])
                print(f"   Avg article save: {avg_article_time:.3f}s")

                # Breakdown
                avg_confidence = sum(a["confidence_scoring"] for a in feed_result["article_times"]) / len(feed_result["article_times"])
                avg_db = sum(a["database_transaction"] for a in feed_result["article_times"]) / len(feed_result["article_times"])
                avg_entity = sum(a["entity_extraction"] for a in feed_result["article_times"]) / len(feed_result["article_times"])
                avg_tag = sum(a["auto_tagging"] for a in feed_result["article_times"]) / len(feed_result["article_times"])

                print(f"     - Confidence scoring: {avg_confidence:.3f}s")
                print(f"     - Database transaction: {avg_db:.3f}s")
                print(f"     - Entity extraction: {avg_entity:.3f}s")
                print(f"     - Auto-tagging: {avg_tag:.3f}s")

            print(f"   Total feed time: {feed_result['total_time']:.2f}s")

        # Analyze results
        results["summary"] = self._analyze_results(results)

        return results

    def _analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze profiling results and identify bottlenecks."""
        summary: Dict[str, Any] = {
            "total_feeds": len(results["feeds"]),
            "total_articles_processed": 0,
            "total_time": 0.0,
            "average_feed_time": 0.0,
            "average_article_time": 0.0,
            "time_breakdown": {},
            "bottlenecks": []
        }

        all_article_times = []
        all_network_times = []
        all_confidence_times = []
        all_db_times = []
        all_entity_times = []

        for feed in results["feeds"]:
            summary["total_time"] += feed["total_time"]
            summary["total_articles_processed"] += len(feed["article_times"])

            all_network_times.append(feed["network_time"])

            for article in feed["article_times"]:
                all_article_times.append(article["total"])
                all_confidence_times.append(article["confidence_scoring"])
                all_db_times.append(article["database_transaction"])
                all_entity_times.append(article["entity_extraction"])

        if all_article_times:
            summary["average_article_time"] = sum(all_article_times) / len(all_article_times)
            summary["average_feed_time"] = summary["total_time"] / max(summary["total_feeds"], 1)

            summary["time_breakdown"] = {
                "network_fetch": {
                    "avg": sum(all_network_times) / len(all_network_times),
                    "percentage": (sum(all_network_times) / summary["total_time"]) * 100
                },
                "confidence_scoring": {
                    "avg": sum(all_confidence_times) / len(all_confidence_times),
                    "percentage": (sum(all_confidence_times) / sum(all_article_times)) * 100
                },
                "database_transaction": {
                    "avg": sum(all_db_times) / len(all_db_times),
                    "percentage": (sum(all_db_times) / sum(all_article_times)) * 100
                },
                "entity_extraction": {
                    "avg": sum(all_entity_times) / len(all_entity_times),
                    "percentage": (sum(all_entity_times) / sum(all_article_times)) * 100
                }
            }

            # Identify bottlenecks (>20% of time is significant)
            for operation, data in summary["time_breakdown"].items():
                if data["percentage"] > 20:
                    summary["bottlenecks"].append({
                        "operation": operation,
                        "avg_time": data["avg"],
                        "percentage": data["percentage"],
                        "severity": "CRITICAL" if data["percentage"] > 40 else "HIGH"
                    })

        return summary

    def print_report(self, results: Dict[str, Any]):
        """Print detailed profiling report."""
        print(f"\n{'='*70}")
        print(f"PROFILING REPORT")
        print(f"{'='*70}\n")

        summary = results["summary"]

        print(f"📊 SUMMARY")
        print(f"   Feeds profiled: {summary['total_feeds']}")
        print(f"   Articles processed: {summary['total_articles_processed']}")
        print(f"   Total time: {summary['total_time']:.2f}s")
        print(f"   Average feed time: {summary['average_feed_time']:.2f}s")
        print(f"   Average article time: {summary['average_article_time']:.3f}s")

        print(f"\n⏱️  TIME BREAKDOWN")
        for operation, data in summary["time_breakdown"].items():
            severity_indicator = ""
            if data["percentage"] > 40:
                severity_indicator = " 🔴 CRITICAL"
            elif data["percentage"] > 20:
                severity_indicator = " 🟡 HIGH"

            print(f"   {operation}:")
            print(f"     Average: {data['avg']:.3f}s")
            print(f"     Percentage: {data['percentage']:.1f}%{severity_indicator}")

        print(f"\n🔍 BOTTLENECKS")
        if summary["bottlenecks"]:
            for bottleneck in sorted(summary["bottlenecks"], key=lambda x: x["percentage"], reverse=True):
                emoji = "🔴" if bottleneck["severity"] == "CRITICAL" else "🟡"
                print(f"   {emoji} {bottleneck['operation']}:")
                print(f"      Average: {bottleneck['avg']:.3f}s ({bottleneck['percentage']:.1f}%)")
                print(f"      Severity: {bottleneck['severity']}")
        else:
            print(f"   ✅ No significant bottlenecks found")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        self._print_recommendations(summary)

    def _print_recommendations(self, summary: Dict[str, Any]):
        """Print optimization recommendations."""
        recommendations = []

        breakdown = summary["time_breakdown"]

        # Network bottleneck
        if breakdown.get("network_fetch", {}).get("percentage", 0) > 30:
            recommendations.append({
                "issue": "Slow network fetches",
                "suggestion": "Consider reducing timeout from 30s or disabling slow feeds",
                "impact": "HIGH"
            })

        # Confidence scoring bottleneck
        if breakdown.get("confidence_scoring", {}).get("percentage", 0) > 20:
            recommendations.append({
                "issue": "Confidence scoring taking too long",
                "suggestion": "Reduce spaCy usage (use_spacy_in_collection config), or cache entity patterns",
                "impact": "HIGH"
            })

        # Database bottleneck
        if breakdown.get("database_transaction", {}).get("percentage", 0) > 30:
            recommendations.append({
                "issue": "Database operations slow",
                "suggestion": "Use batch inserts instead of per-article transactions, or reduce WAL mode overhead",
                "impact": "MEDIUM"
            })

        # Entity extraction bottleneck
        if breakdown.get("entity_extraction", {}).get("percentage", 0) > 15:
            recommendations.append({
                "issue": "Entity extraction per article",
                "suggestion": "Disable auto-tagging during collection (run as batch job instead)",
                "impact": "HIGH"
            })

        if recommendations:
            for rec in sorted(recommendations, key=lambda x: x["impact"] == "HIGH", reverse=True):
                print(f"   ⚠️  {rec['issue']}")
                print(f"      Suggestion: {rec['suggestion']}")
                print(f"      Impact: {rec['impact']}")
        else:
            print(f"   ✅ Collection performance is optimal")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Profile RSS collection performance")
    parser.add_argument("--num-feeds", type=int, default=5, help="Number of feeds to profile (default: 5)")
    parser.add_argument("--output-report", action="store_true", help="Save report to JSON file")
    parser.add_argument("--profile", action="store_true", help="Run with cProfile for detailed analysis")

    args = parser.parse_args()

    profiler = CollectionProfiler()

    if args.profile:
        # Run with cProfile for detailed function-level profiling
        print("Running with cProfile...")
        profiler_out = StringIO()
        cp = cProfile.Profile()

        cp.enable()
        results = profiler.profile_collection(num_feeds=args.num_feeds)
        cp.disable()

        profiler.print_report(results)

        # Print cProfile stats
        print(f"\n{'='*70}")
        print(f"CPROFILE DETAILED ANALYSIS")
        print(f"{'='*70}\n")

        stats = pstats.Stats(cp, stream=profiler_out)
        stats.sort_stats("cumulative")
        stats.print_stats(20)  # Top 20 functions

        print(profiler_out.getvalue())
    else:
        results = profiler.profile_collection(num_feeds=args.num_feeds)
        profiler.print_report(results)

    # Save report if requested
    if args.output_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"profiling_report_{timestamp}.json"

        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
