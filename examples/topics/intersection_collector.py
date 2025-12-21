#!/usr/bin/env python3
"""
Topic Intersection Collector for AI News

This script allows you to collect news articles that contain specific
topic intersections using the advanced intersection optimization system.

Usage:
    python topic_intersection_collector.py --topics "AI,Healthcare" --topics "AI,Finance"
    python topic_intersection_collector.py --config "AI+Healthcare" "Blockchain+Finance" 
    python topic_intersection_collector.py --list-presets
"""

import argparse
import json
import sys
from typing import List, Dict, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_news.config import Config
from ai_news.database import Database
from ai_news.collector import SimpleCollector
from ai_news.intersection_optimization import create_intersection_optimizer
from ai_news.search_collector import SearchEngineCollector


# Predefined topic intersection presets
TOPIC_PRESETS = {
    "AI+Healthcare": {
        "topics": ["AI", "Healthcare", "Medical", "Machine Learning"],
        "description": "AI applications in healthcare and medical diagnosis",
        "min_confidence": 0.3
    },
    "AI+Finance": {
        "topics": ["AI", "Finance", "Banking", "Fintech", "Trading"],
        "description": "AI applications in financial services and trading",
        "min_confidence": 0.3
    },
    "Blockchain+Finance": {
        "topics": ["Blockchain", "Cryptocurrency", "Finance", "Banking"],
        "description": "Blockchain and cryptocurrency in financial services",
        "min_confidence": 0.25
    },
    "AI+Automotive": {
        "topics": ["AI", "Autonomous", "Self-driving", "Automotive", "Cars"],
        "description": "AI in autonomous vehicles and automotive industry",
        "min_confidence": 0.3
    },
    "AI+Education": {
        "topics": ["AI", "Education", "Learning", "EdTech", "Students"],
        "description": "AI applications in education and learning systems",
        "min_confidence": 0.25
    },
    "Cybersecurity+AI": {
        "topics": ["Cybersecurity", "AI", "Security", "Threat Detection"],
        "description": "AI applications in cybersecurity and threat detection",
        "min_confidence": 0.3
    }
}


class TopicIntersectionCollector:
    """Collect news articles based on topic intersections."""
    
    def __init__(self, db_path: str = "ai_news.db"):
        """Initialize the topic intersection collector."""
        self.config = Config()
        self.database = Database(db_path)
        self.collector = SimpleCollector(self.database)
        self.search_collector = SearchEngineCollector(self.database)
        self.optimizer = create_intersection_optimizer()
        
        print("🐶 Topic Intersection Collector initialized!")
        print(f"   Database: {db_path}")
        print(f"   Available presets: {len(TOPIC_PRESETS)}")
    
    def collect_topic_intersections(
        self, 
        topic_groups: List[List[str]], 
        min_confidence: float = 0.25,
        use_search: bool = True,
        max_articles_per_group: int = 20
    ) -> Dict[str, Any]:
        """
        Collect articles for multiple topic intersection groups.
        
        Args:
            topic_groups: List of topic lists to find intersections for
            min_confidence: Minimum confidence score for intersections
            use_search: Whether to use search engine collection
            max_articles_per_group: Maximum articles to collect per topic group
            
        Returns:
            Collection results with intersection analysis
        """
        results = {
            "topic_groups": [],
            "total_articles_collected": 0,
            "intersection_articles": [],
            "collection_time": datetime.now().isoformat(),
            "stats": {}
        }
        
        print(f"\n🎯 Starting collection for {len(topic_groups)} topic groups...")
        print(f"   Minimum confidence: {min_confidence}")
        print(f"   Using search engines: {use_search}")
        
        for i, topics in enumerate(topic_groups, 1):
            print(f"\n[{i}/{len(topic_groups)}] Processing topic group: {' + '.join(topics)}")
            
            # Collect articles for this topic group
            group_result = self._collect_single_topic_group(
                topics, min_confidence, use_search, max_articles_per_group
            )
            
            results["topic_groups"].append({
                "topics": topics,
                "result": group_result
            })
            
            results["total_articles_collected"] += group_result["total_collected"]
            results["intersection_articles"].extend(group_result["intersection_articles"])
            
            print(f"   ✅ Collected {group_result['intersection_found']} intersection articles")
        
        # Calculate overall stats
        results["stats"] = self._calculate_overall_stats(results)
        
        return results
    
    def _collect_single_topic_group(
        self,
        topics: List[str],
        min_confidence: float,
        use_search: bool,
        max_articles: int
    ) -> Dict[str, Any]:
        """Collect articles for a single topic group."""
        
        # Create search query from topics
        search_query = " AND ".join(topics)
        print(f"   🔍 Search query: {search_query}")
        
        group_result = {
            "topics": topics,
            "search_query": search_query,
            "total_collected": 0,
            "intersection_found": 0,
            "intersection_articles": [],
            "relevance_scores": [],
            "collection_method": "search" if use_search else "feeds"
        }
        
        # Collect articles using search engine
        if use_search:
            try:
                # Try DuckDuckGo search first
                search_results = self.search_collector.search_duckduckgo(
                    search_query, max_results=max_articles
                )
                
                print(f"   📊 Found {len(search_results)} search results")
                
                # Process search results
                for result in search_results:
                    article_data = self._convert_search_result_to_article(result)
                    if article_data:
                        # Check for topic intersections
                        intersection_analysis = self.optimizer.detect_weighted_intersections(
                            article_data, topics
                        )
                        
                        if intersection_analysis["intersection_detected"]:
                            # Validate relevance
                            validation = self.optimizer.validate_intersection_relevance(
                                intersection_analysis, article_data
                            )
                            
                            if validation["is_relevant"] and intersection_analysis["confidence"] >= min_confidence:
                                # Save to database
                                from ai_news.database import Article
                                article = Article(
                                    title=article_data.get("title", ""),
                                    url=article_data.get("url", ""),
                                    summary=article_data.get("summary", ""),
                                    content=article_data.get("content", ""),
                                    source_name=article_data.get("source", "Search"),
                                    ai_relevant=True,
                                    category=f"Intersection: {' + '.join(topics)}"
                                )
                                
                                if self.database.save_article(article):
                                    group_result["intersection_articles"].append({
                                        "title": article.title,
                                        "confidence": intersection_analysis["confidence"],
                                        "relevance": validation["relevance_score"],
                                        "matches": len(intersection_analysis.get("matches", []))
                                    })
                                    group_result["intersection_found"] += 1
                                    group_result["relevance_scores"].append(validation["relevance_score"])
                        
                        group_result["total_collected"] += 1
                
            except Exception as e:
                print(f"   ⚠️  Search collection failed: {e}")
                print("   🔄 Falling back to feed collection...")
                use_search = False
        
        # Fallback to RSS feeds if search fails
        if not use_search:
            # Collect from all feeds and filter for intersections
            feed_configs = self.config.get_feed_configs()
            feed_stats = self.collector.collect_all_feeds(feed_configs)
            
            # Get recent articles and check for intersections
            recent_articles = self.database.get_recent_articles(days=1, limit=100)
            
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
                        group_result["intersection_articles"].append({
                            "title": article.title,
                            "confidence": intersection_analysis["confidence"],
                            "relevance": validation["relevance_score"],
                            "matches": len(intersection_analysis.get("matches", []))
                        })
                        group_result["intersection_found"] += 1
                        group_result["relevance_scores"].append(validation["relevance_score"])
                
                group_result["total_collected"] += 1
                
                # Stop if we've found enough intersections
                if group_result["intersection_found"] >= max_articles:
                    break
        
        return group_result
    
    def _convert_search_result_to_article(self, result: Dict) -> Dict[str, Any]:
        """Convert search result to article data format."""
        return {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "summary": result.get("snippet", ""),
            "content": result.get("snippet", ""),  # Use snippet as content for search results
            "source": result.get("source", "Search Engine")
        }
    
    def _calculate_overall_stats(self, results: Dict) -> Dict[str, Any]:
        """Calculate overall collection statistics."""
        stats = {
            "total_topic_groups": len(results["topic_groups"]),
            "total_articles_analyzed": results["total_articles_collected"],
            "total_intersection_articles": len(results["intersection_articles"]),
            "average_articles_per_group": 0,
            "average_confidence": 0.0,
            "high_quality_intersections": 0,
            "topic_groups_with_matches": 0
        }
        
        if results["topic_groups"]:
            stats["average_articles_per_group"] = (
                results["total_articles_collected"] / len(results["topic_groups"])
            )
        
        # Calculate confidence averages and high-quality intersections
        all_confidences = []
        all_relevance_scores = []
        
        for group in results["topic_groups"]:
            group_result = group["result"]
            if group_result["intersection_found"] > 0:
                stats["topic_groups_with_matches"] += 1
                all_confidences.extend([a["confidence"] for a in group_result["intersection_articles"]])
                all_relevance_scores.extend([a["relevance"] for a in group_result["intersection_articles"]])
                
                # Count high-quality intersections (confidence > 0.5, relevance > 0.7)
                high_quality = sum(
                    1 for a in group_result["intersection_articles"]
                    if a["confidence"] > 0.5 and a["relevance"] > 0.7
                )
                stats["high_quality_intersections"] += high_quality
        
        if all_confidences:
            stats["average_confidence"] = sum(all_confidences) / len(all_confidences)
            stats["average_relevance"] = sum(all_relevance_scores) / len(all_relevance_scores)
        
        return stats
    
    def print_results(self, results: Dict):
        """Print formatted collection results."""
        print("\n" + "="*70)
        print("🎯 TOPIC INTERSECTION COLLECTION RESULTS")
        print("="*70)
        
        stats = results["stats"]
        print(f"📊 Summary:")
        print(f"   Topic groups processed: {stats['total_topic_groups']}")
        print(f"   Articles analyzed: {stats['total_articles_analyzed']}")
        print(f"   Intersection articles found: {stats['total_intersection_articles']}")
        print(f"   Groups with matches: {stats['topic_groups_with_matches']}/{stats['total_topic_groups']}")
        print(f"   High-quality intersections: {stats['high_quality_intersections']}")
        
        if stats.get('average_confidence', 0) > 0:
            print(f"   Average confidence: {stats['average_confidence']:.3f}")
            print(f"   Average relevance: {stats['average_relevance']:.3f}")
        
        print(f"\n🔍 Details by Topic Group:")
        for i, group in enumerate(results["topic_groups"], 1):
            topics = " + ".join(group["topics"])
            result = group["result"]
            
            print(f"\n{i}. {topics}")
            print(f"   Articles analyzed: {result['total_collected']}")
            print(f"   Intersections found: {result['intersection_found']}")
            print(f"   Collection method: {result['collection_method']}")
            
            if result["intersection_articles"]:
                print(f"   Top intersection articles:")
                for j, article in enumerate(result["intersection_articles"][:3], 1):
                    print(f"     {j}. {article['title'][:80]}...")
                    print(f"        Confidence: {article['confidence']:.3f}, Relevance: {article['relevance']:.3f}")
                
                if len(result["intersection_articles"]) > 3:
                    print(f"     ... and {len(result['intersection_articles']) - 3} more")
        
        print(f"\n✨ Collection completed at: {results['collection_time']}")


def parse_topic_groups(args) -> List[List[str]]:
    """Parse topic groups from command line arguments."""
    topic_groups = []
    
    # Handle preset configurations
    if hasattr(args, 'preset') and args.preset:
        for preset_name in args.preset:
            if preset_name in TOPIC_PRESETS:
                preset = TOPIC_PRESETS[preset_name]
                topic_groups.append(preset["topics"])
                print(f"✅ Using preset: {preset_name}")
                print(f"   {preset['description']}")
            else:
                print(f"⚠️  Unknown preset: {preset_name}")
                print(f"   Available presets: {', '.join(TOPIC_PRESETS.keys())}")
    
    # Handle custom topic combinations
    if hasattr(args, 'topics') and args.topics:
        for topic_string in args.topics:
            topics = [t.strip() for t in topic_string.split(',')]
            if len(topics) >= 2:  # Need at least 2 topics for intersection
                topic_groups.append(topics)
                print(f"✅ Custom topic group: {' + '.join(topics)}")
            else:
                print(f"⚠️  Need at least 2 topics for intersection: {topic_string}")
    
    return topic_groups


def list_presets():
    """List all available topic presets."""
    print("\n🎯 Available Topic Intersection Presets:")
    print("="*50)
    
    for name, preset in TOPIC_PRESETS.items():
        print(f"\n{name}:")
        print(f"   Description: {preset['description']}")
        print(f"   Topics: {' + '.join(preset['topics'])}")
        print(f"   Min confidence: {preset['min_confidence']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect AI news articles for specific topic intersections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use preset configurations
  python topic_intersection_collector.py --preset "AI+Healthcare" "AI+Finance"
  
  # Custom topic combinations
  python topic_intersection_collector.py --topics "AI,Healthcare,Medical" --topics "Blockchain,Cryptocurrency,Finance"
  
  # List available presets
  python topic_intersection_collector.py --list-presets
  
  # Custom confidence threshold
  python topic_intersection_collector.py --preset "AI+Healthcare" --min-confidence 0.4
        """
    )
    
    parser.add_argument(
        "--preset", "-p",
        nargs="+",
        help="Use preset topic intersection configurations"
    )
    
    parser.add_argument(
        "--topics", "-t",
        nargs="+",
        help="Custom topic combinations (comma-separated, multiple groups supported)"
    )
    
    parser.add_argument(
        "--list-presets", "-l",
        action="store_true",
        help="List all available preset configurations"
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
        "--no-search",
        action="store_true",
        help="Skip search engine collection, use RSS feeds only"
    )
    
    parser.add_argument(
        "--db-path",
        default="ai_news.db",
        help="Database file path (default: ai_news.db)"
    )
    
    args = parser.parse_args()
    
    # Handle list presets
    if args.list_presets:
        list_presets()
        return
    
    # Validate arguments
    if not args.preset and not args.topics:
        print("❌ Please specify either --preset or --topics")
        print("   Use --list-presets to see available configurations")
        parser.print_help()
        return
    
    # Parse topic groups
    topic_groups = parse_topic_groups(args)
    
    if not topic_groups:
        print("❌ No valid topic groups specified")
        return
    
    # Initialize collector
    collector = TopicIntersectionCollector(args.db_path)
    
    # Collect articles for topic intersections
    try:
        results = collector.collect_topic_intersections(
            topic_groups=topic_groups,
            min_confidence=args.min_confidence,
            use_search=not args.no_search,
            max_articles_per_group=args.max_articles
        )
        
        # Print results
        collector.print_results(results)
        
        # Save results to file
        output_file = f"topic_intersections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()