#!/usr/bin/env python3
"""
Arbitrary Topic Collector

A simple CLI tool to collect news for ANY topic combination without preconfiguration.
This bridges the gap between the existing AI news system and user needs.

Usage:
    python arbitrary_topic_collector.py "quantum computing" "blockchain healthcare"
    python arbitrary_topic_collector.py --topics "renewable energy" "AI ethics" --min-confidence 0.3
    python arbitrary_topic_collector.py --interactive
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_news.intersection_optimization import create_intersection_optimizer
from ai_news.search_collector import SearchEngineCollector
from ai_news.database import Database
from ai_news.collector import SimpleCollector
from ai_news.config import Config


class ArbitraryTopicCollector:
    """Collect news for arbitrary topic combinations."""
    
    def __init__(self, db_path: str = "ai_news.db"):
        """Initialize the collector."""
        self.config = Config()
        self.database = Database(db_path)
        self.collector = SimpleCollector(self.database)
        self.search_collector = SearchEngineCollector(self.database)
        self.optimizer = create_intersection_optimizer()
        
        print("🎯 Arbitrary Topic Collector")
        print("✅ Works with ANY topic combination - no preconfiguration needed!")
        print(f"📁 Database: {db_path}")
    
    def collect_topic_intersections(
        self, 
        topic_groups: List[List[str]], 
        min_confidence: float = 0.25,
        max_articles: int = 20
    ) -> Dict[str, Any]:
        """Collect articles for multiple topic intersection groups."""
        results = {
            "topic_groups": [],
            "total_articles_collected": 0,
            "intersection_articles": [],
            "collection_time": datetime.now().isoformat(),
            "success_rate": 0.0
        }
        
        print(f"\n🚀 Collecting {len(topic_groups)} topic intersection groups...")
        print(f"📊 Minimum confidence: {min_confidence}")
        print(f"🔢 Max articles per group: {max_articles}")
        
        successful_groups = 0
        
        for i, topics in enumerate(topic_groups, 1):
            print(f"\n[{i}/{len(topic_groups)}] Processing: {' + '.join(topics)}")
            
            group_result = self._collect_single_topic_group(topics, min_confidence, max_articles)
            
            if group_result["success"]:
                successful_groups += 1
                print(f"   ✅ SUCCESS: Found {group_result['articles_found']} intersection articles")
            else:
                print(f"   ⚠️  LIMITED: Found {group_result['articles_found']} articles (may need adjustment)")
            
            results["topic_groups"].append({
                "topics": topics,
                "result": group_result
            })
            
            results["total_articles_collected"] += group_result["articles_analyzed"]
            results["intersection_articles"].extend(group_result["intersection_articles"])
        
        results["success_rate"] = successful_groups / len(topic_groups)
        
        return results
    
    def _collect_single_topic_group(
        self, 
        topics: List[str], 
        min_confidence: float, 
        max_articles: int
    ) -> Dict[str, Any]:
        """Collect articles for a single topic group."""
        group_result = {
            "topics": topics,
            "articles_analyzed": 0,
            "articles_found": 0,
            "intersection_articles": [],
            "success": False,
            "collection_method": ""
        }
        
        # Method 1: Search engine collection (preferred)
        try:
            print(f"   🔍 Trying search engine collection...")
            search_articles = self._collect_from_search_engine(topics, max_articles)
            
            if search_articles:
                group_result["articles_analyzed"] = len(search_articles)
                group_result["intersection_articles"] = search_articles
                group_result["articles_found"] = len(search_articles)
                group_result["success"] = len(search_articles) > 0
                group_result["collection_method"] = "search_engine"
                
                print(f"   📊 Search engine: {len(search_articles)} articles found")
                return group_result
            else:
                print(f"   ⚠️  Search engine returned no results")
        
        except Exception as e:
            print(f"   ⚠️  Search engine failed: {e}")
        
        # Method 2: Database analysis (fallback)
        try:
            print(f"   🔄 Fallback: Analyzing existing database...")
            db_articles = self._collect_from_database(topics, min_confidence)
            
            if db_articles:
                group_result["articles_analyzed"] = db_articles["analyzed_count"]
                group_result["intersection_articles"] = db_articles["intersection_articles"]
                group_result["articles_found"] = len(db_articles["intersection_articles"])
                group_result["success"] = len(db_articles["intersection_articles"]) > 0
                group_result["collection_method"] = "database_fallback"
                
                print(f"   📊 Database fallback: {len(db_articles['intersection_articles'])} articles found")
                return group_result
            else:
                print(f"   ⚠️  Database fallback found no results")
        
        except Exception as e:
            print(f"   ⚠️  Database fallback failed: {e}")
        
        # Method 3: RSS feed collection (last resort)
        try:
            print(f"   🔄 Last resort: RSS feed collection...")
            rss_result = self._collect_from_rss_feeds(topics, min_confidence, max_articles)
            
            group_result["articles_analyzed"] = rss_result["analyzed_count"]
            group_result["intersection_articles"] = rss_result["intersection_articles"]
            group_result["articles_found"] = len(rss_result["intersection_articles"])
            group_result["success"] = len(rss_result["intersection_articles"]) > 0
            group_result["collection_method"] = "rss_feeds"
            
            print(f"   📊 RSS feeds: {len(rss_result['intersection_articles'])} articles found")
            
        except Exception as e:
            print(f"   ⚠️  RSS feed collection failed: {e}")
        
        return group_result
    
    def _collect_from_search_engine(self, topics: List[str], max_articles: int) -> List[Dict]:
        """Collect articles from search engines."""
        query = " AND ".join(topics)
        search_results = self.search_collector.search_duckduckgo(query, max_results=max_articles * 2)
        
        intersection_articles = []
        
        for result in search_results:
            article_data = {
                "title": result.get("title", ""),
                "content": result.get("snippet", ""),
                "summary": result.get("snippet", "")
            }
            
            # Check for topic intersections
            intersection_analysis = self.optimizer.detect_weighted_intersections(
                article_data, topics
            )
            
            if intersection_analysis["intersection_detected"]:
                validation = self.optimizer.validate_intersection_relevance(
                    intersection_analysis, article_data
                )
                
                if validation["is_relevant"]:
                    intersection_articles.append({
                        "title": article_data["title"],
                        "url": result.get("url", ""),
                        "source": result.get("source", "Search Engine"),
                        "confidence": intersection_analysis["confidence"],
                        "relevance": validation["relevance_score"],
                        "summary": article_data["summary"]
                    })
        
        return intersection_articles
    
    def _collect_from_database(self, topics: List[str], min_confidence: float) -> Dict[str, Any]:
        """Collect articles by analyzing existing database."""
        # Get recent articles from database
        recent_articles = self.database.get_articles(limit=200, ai_only=False)
        
        intersection_articles = []
        
        for article in recent_articles:
            article_data = {
                "title": article.title,
                "content": article.content or "",
                "summary": article.summary
            }
            
            intersection_analysis = self.optimizer.detect_weighted_intersections(
                article_data, topics
            )
            
            if (intersection_analysis["intersection_detected"] and 
                intersection_analysis["confidence"] >= min_confidence):
                
                validation = self.optimizer.validate_intersection_relevance(
                    intersection_analysis, article_data
                )
                
                if validation["is_relevant"]:
                    intersection_articles.append({
                        "title": article.title,
                        "url": article.url,
                        "source": article.source_name,
                        "confidence": intersection_analysis["confidence"],
                        "relevance": validation["relevance_score"],
                        "summary": article.summary
                    })
        
        return {
            "analyzed_count": len(recent_articles),
            "intersection_articles": intersection_articles
        }
    
    def _collect_from_rss_feeds(self, topics: List[str], min_confidence: float, max_articles: int) -> Dict[str, Any]:
        """Collect articles from RSS feeds and filter for intersections."""
        # Collect from all RSS feeds (using a simple approach)
        try:
            # Use existing feeds from config if available
            feed_stats = self.collector.collect_all_feeds([])  # Empty list will use defaults
        except:
            # If that fails, we can't collect from RSS
            return {"analyzed_count": 0, "intersection_articles": []}
        
        # Get fresh articles and analyze
        fresh_articles = self.database.get_articles(limit=100, ai_only=False)
        
        intersection_articles = []
        
        for article in fresh_articles:
            article_data = {
                "title": article.title,
                "content": article.content or "",
                "summary": article.summary
            }
            
            intersection_analysis = self.optimizer.detect_weighted_intersections(
                article_data, topics
            )
            
            if (intersection_analysis["intersection_detected"] and 
                intersection_analysis["confidence"] >= min_confidence):
                
                validation = self.optimizer.validate_intersection_relevance(
                    intersection_analysis, article_data
                )
                
                if validation["is_relevant"]:
                    intersection_articles.append({
                        "title": article.title,
                        "url": article.url,
                        "source": article.source_name,
                        "confidence": intersection_analysis["confidence"],
                        "relevance": validation["relevance_score"],
                        "summary": article.summary
                    })
            
            # Stop if we've found enough
            if len(intersection_articles) >= max_articles:
                break
        
        return {
            "analyzed_count": len(fresh_articles),
            "intersection_articles": intersection_articles
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted collection results."""
        print("\n" + "="*70)
        print("🎯 ARBITRARY TOPIC COLLECTION RESULTS")
        print("="*70)
        
        success_rate = results["success_rate"]
        print(f"\n📊 Summary:")
        print(f"   Topic groups processed: {len(results['topic_groups'])}")
        print(f"   Articles analyzed: {results['total_articles_collected']}")
        print(f"   Intersection articles found: {len(results['intersection_articles'])}")
        print(f"   Success rate: {success_rate:.1%}")
        
        if success_rate >= 0.8:
            print(f"   ✅ EXCELLENT: Most topic combinations worked!")
        elif success_rate >= 0.5:
            print(f"   ✅ GOOD: Many topic combinations worked")
        else:
            print(f"   ⚠️  LIMITED: Consider adjusting confidence or topics")
        
        print(f"\n🔍 Results by Topic Group:")
        for i, group in enumerate(results["topic_groups"], 1):
            topics = " + ".join(group["topics"])
            result = group["result"]
            
            print(f"\n{i}. {topics}")
            print(f"   Method: {result['collection_method']}")
            print(f"   Articles found: {result['articles_found']}")
            
            if result["intersection_articles"]:
                print(f"   Top articles:")
                for j, article in enumerate(result["intersection_articles"][:3], 1):
                    print(f"     {j}. {article['title'][:70]}...")
                    print(f"        📊 Confidence: {article['confidence']:.3f}, Relevance: {article['relevance']:.3f}")
                    print(f"        📰 Source: {article['source']}")
                
                if len(result["intersection_articles"]) > 3:
                    print(f"     ... and {len(result['intersection_articles']) - 3} more")
        
        print(f"\n💾 Collection completed at: {results['collection_time']}")
        
        # Save results
        output_file = f"arbitrary_topics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def interactive_mode(self):
        """Run interactive collection mode."""
        print("\n🎮 INTERACTIVE MODE")
        print("Enter topic combinations (comma-separated) to collect news.")
        print("Type 'quit' to exit, 'help' for examples.\n")
        
        while True:
            try:
                user_input = input("🔍 Enter topics (comma-separated): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\n💡 Examples:")
                    print("   • quantum computing,healthcare")
                    print("   • blockchain,finance,insurance")
                    print("   • renewable energy,AI,storage")
                    print("   • CRISPR,gene editing,ethics")
                    print("   • supply chain,automation,AI")
                    print("")
                    continue
                
                if not user_input:
                    continue
                
                # Parse topics
                topics = [t.strip() for t in user_input.split(',') if t.strip()]
                
                if len(topics) < 2:
                    print("⚠️  Please enter at least 2 topics separated by commas")
                    continue
                
                print(f"\n🚀 Collecting: {' + '.join(topics)}")
                
                # Quick collection
                results = self.collect_topic_intersections([topics])
                self.print_results(results)
                
                print("\n" + "-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect news for ANY topic combination without preconfiguration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect for custom topic intersections
  python arbitrary_topic_collector.py "quantum computing" "blockchain healthcare"
  
  # Single topic group with confidence threshold
  python arbitrary_topic_collector.py --topics "renewable energy,AI" --min-confidence 0.3
  
  # Interactive mode
  python arbitrary_topic_collector.py --interactive
  
  # High-sensitivity search
  python arbitrary_topic_collector.py --topics "web3,crypto" --min-confidence 0.1 --max-articles 50
        """
    )
    
    parser.add_argument(
        "topics", 
        nargs="*",
        help="Topic combinations (comma-separated within each group, multiple groups supported)"
    )
    
    parser.add_argument(
        "--topics-list", "-t",
        nargs="+",
        help="Alternative way to specify topic combinations"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--min-confidence", "-c",
        type=float,
        default=0.25,
        help="Minimum confidence score for intersections (default: 0.25)"
    )
    
    parser.add_argument(
        "--max-articles", "-m",
        type=int,
        default=20,
        help="Maximum articles to collect per topic group (default: 20)"
    )
    
    parser.add_argument(
        "--db-path",
        default="ai_news.db",
        help="Database file path (default: ai_news.db)"
    )
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = ArbitraryTopicCollector(args.db_path)
    
    if args.interactive:
        collector.interactive_mode()
        return
    
    # Parse topic groups
    topic_groups = []
    
    if args.topics:
        for topic_string in args.topics:
            topics = [t.strip() for t in topic_string.split(',') if t.strip()] 
            if len(topics) >= 2:
                topic_groups.append(topics)
            else:
                print(f"⚠️  Skipping '{topic_string}' - need at least 2 topics")
    
    elif args.topics_list:
        for topic_string in args.topics_list:
            topics = [t.strip() for t in topic_string.split(',') if t.strip()] 
            if len(topics) >= 2:
                topic_groups.append(topics)
            else:
                print(f"⚠️  Skipping '{topic_string}' - need at least 2 topics")
    
    else:
        # Default demo topics
        topic_groups = [
            ["Quantum", "Computing"],
            ["AI", "Healthcare"],
            ["Blockchain", "Finance"]
        ]
        print("\n🔧 No topics specified, using demo topics...")
    
    if not topic_groups:
        print("❌ No valid topic groups provided!")
        print("   Use --topics, --interactive, or run without arguments for demo")
        return
    
    # Collect articles
    try:
        results = collector.collect_topic_intersections(
            topic_groups=topic_groups,
            min_confidence=args.min_confidence,
            max_articles=args.max_articles
        )
        
        collector.print_results(results)
        
    except KeyboardInterrupt:
        print("\n⚠️  Collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()