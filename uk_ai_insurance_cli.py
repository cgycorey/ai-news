#!/usr/bin/env python3
"""
UK AI Insurance Collection and Search CLI Tool
Leverages enhanced multi-keyword matching for comprehensive AI + insurance discovery
"""

import argparse
import sys
import os
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_news.database import Database
from ai_news.collector import SimpleCollector
from ai_news.config import Config
from comprehensive_search import FlexibleSearchEngine

class UKAIInsuranceCLI:
    """CLI tool for UK AI insurance collection and analysis."""
    
    def __init__(self):
        # Load config properly from file
        from pathlib import Path
        config_path = Path("config.json")
        self.config = Config.load(config_path)
        self.db = Database("ai_news.db")
        self.collector = SimpleCollector(database=self.db)
        self.search_engine = FlexibleSearchEngine(self.db)
        
        # Enhanced keyword combinations for AI + insurance
        self.ai_insurance_combinations = [
            # Primary AI terms
            ["AI", "artificial intelligence", "machine learning", "deep learning", "LLM", "GPT", "ChatGPT"],
            # Insurance terms  
            ["insurance", "insurtech", "underwriting", "claims", "risk", "premium", "coverage"],
            # Fintech/Regtech overlap
            ["fintech", "regtech", "embedded finance", "digital banking"],
            # Advanced AI applications
            ["predictive analytics", "fraud detection", "automation", "algorithmic"],
            # Business context
            ["compliance", "assessment", "modeling", "forecasting"]
        ]
    
    def collect_uk_ai_insurance(self, limit=None, force=False):
        """Collect UK AI insurance content with enhanced keyword matching."""
        print("🇬🇧 Starting UK AI Insurance Collection")
        print("=" * 60)
        
        # Get UK feeds - use the same method as CLI
        uk_feeds = []
        try:
            # Access regions like the CLI does
            regions = self.config.regions
            if 'uk' in regions:
                uk_region = regions['uk']
                uk_feeds = uk_region.feeds
        except Exception as e:
            print(f"   ⚠️  Could not access UK feeds: {e}")
            # Try manual feed loading as fallback
            try:
                import json
                with open('config.json', 'r') as f:
                    config_data = json.load(f)
                    uk_region_data = config_data.get('regions', {}).get('uk', {})
                    feeds_data = uk_region_data.get('feeds', [])
                    from ai_news.config import FeedConfig
                    uk_feeds = [FeedConfig.from_dict(feed) for feed in feeds_data]
            except Exception as e2:
                print(f"   ⚠️  Fallback loading failed: {e2}")
        
        print(f"📡 Found {len(uk_feeds)} UK feeds")
        
        # Show feed status
        for i, feed in enumerate(uk_feeds, 1):
            status = "✅" if feed.enabled else "❌"
            print(f"  {status} {feed.name} ({feed.category})")
        
        if not force:
            # Check recent collection
            recent_articles = self.db.get_articles(limit=10, region='uk')
            if recent_articles:
                latest_date = max(a.published_at for a in recent_articles if a.published_at)
                hours_old = (datetime.now() - latest_date.replace(tzinfo=None)).total_seconds() / 3600
                if hours_old < 2:
                    print(f"\n⏰ Recent collection found ({hours_old:.1f} hours old)")
                    response = input("Force new collection? (y/N): ")
                    if response.lower() != 'y':
                        print("Skipping collection...")
                        return
        
        print(f"\n🔄 Collecting UK news with enhanced AI + insurance detection...")
        
        # Collect with enhanced keyword matching
        total_articles = 0
        ai_relevant = 0
        ai_insurance = 0
        
        for feed in uk_feeds:
            if not feed.enabled:
                continue
                
            print(f"  📡 Processing {feed.name}...")
            try:
                articles = self.collector.fetch_feed(feed, max_articles=limit or 50)
                total_articles += len(articles)
                
                # Analyze collected articles
                for article in articles:
                    if article.ai_relevant:
                        ai_relevant += 1
                        
                        # Check for insurance intersection
                        content_text = (article.title + " " + article.content).lower()
                        insurance_keywords = ['insurance', 'insurtech', 'underwriting', 'claims', 'risk', 'premium']
                        if any(kw in content_text for kw in insurance_keywords):
                            ai_insurance += 1
                
                print(f"    ✅ {len(articles)} articles collected")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
        
        print(f"\n📊 Collection Summary:")
        print(f"   Total Articles: {total_articles}")
        print(f"   AI-Relevant: {ai_relevant}")
        print(f"   AI + Insurance: {ai_insurance}")
        print(f"   AI + Insurance Rate: {(ai_insurance/ai_relevant*100):.1f}%" if ai_relevant > 0 else "   AI + Insurance Rate: 0%")
    
    def search_uk_ai_insurance(self, query=None, strategy="smart", limit=10, show_details=False):
        """Search UK AI insurance content with enhanced keyword combinations."""
        print("🔍 UK AI Insurance Search")
        print("=" * 60)
        
        if not query:
            # Use comprehensive AI + insurance search
            queries = [
                "AI insurance",
                "artificial intelligence insurance", 
                "machine learning insurance",
                "insurtech AI",
                "insurance technology AI",
                "risk assessment AI",
                "underwriting AI",
                "claims processing AI",
                "predictive analytics insurance",
                "fraud detection AI insurance"
            ]
            
            print("🔄 Running comprehensive AI + insurance search...")
            all_results = []
            
            for search_query in queries:
                print(f"  🔍 Searching: '{search_query}'")
                results = self.search_engine.search(
                    search_query, 
                    strategy=strategy,
                    region='uk',
                    ai_only=True,
                    limit=limit
                )
                
                # Deduplicate by URL
                seen_urls = set(r.url for r in all_results)
                new_results = [r for r in results if r.url not in seen_urls]
                all_results.extend(new_results)
                
                if new_results:
                    print(f"    ✅ Found {len(new_results)} new articles")
            
            results = all_results[:limit]
            
        else:
            # Custom query search
            print(f"🔍 Searching: '{query}'")
            results = self.search_engine.search(
                query,
                strategy=strategy,
                region='uk',
                ai_only=True,
                limit=limit
            )
        
        if not results:
            print("❌ No UK AI insurance articles found")
            return
        
        print(f"\n📊 Found {len(results)} UK AI insurance articles:")
        print("=" * 80)
        
        for i, article in enumerate(results, 1):
            # Article header
            relevance_indicator = "🤖" if article.ai_relevant else "  "
            score_text = f"[Score: {article.search_score:02d}]" if hasattr(article, 'search_score') else ""
            
            print(f"{i}. {relevance_indicator} {article.title} {score_text}")
            print(f"   📰 {article.source_name} | 📅 {article.published_at.strftime('%Y-%m-%d') if article.published_at else 'Unknown'}")
            
            # Enhanced keyword analysis
            content_text = (article.title + " " + article.content).lower()
            
            # AI keywords found
            ai_keywords = []
            if article.ai_keywords_found:
                ai_keywords = article.ai_keywords_found
            
            # Insurance keywords found
            insurance_keywords = ['insurance', 'insurtech', 'underwriting', 'claims', 'risk', 'premium', 'coverage']
            found_insurance = [kw for kw in insurance_keywords if kw in content_text]
            
            # Fintech keywords found
            fintech_keywords = ['fintech', 'regtech', 'embedded finance', 'digital banking']
            found_fintech = [kw for kw in fintech_keywords if kw in content_text]
            
            print(f"   🤖 AI Keywords: {', '.join(ai_keywords) if ai_keywords else 'None'}")
            print(f"   🛡️  Insurance: {', '.join(found_insurance) if found_insurance else 'None'}")
            print(f"   💰 Fintech: {', '.join(found_fintech) if found_fintech else 'None'}")
            
            # Search details
            if show_details and hasattr(article, 'search_details'):
                print(f"   🔍 Search: {', '.join(article.search_details)}")
            
            # Content snippet
            content_preview = article.content[:200] + "..." if len(article.content) > 200 else article.content
            print(f"   📄 {content_preview}")
            print(f"   🔗 {article.url}")
            print()
    
    def analyze_uk_ai_insurance(self):
        """Analyze UK AI insurance landscape with comprehensive insights."""
        print("📈 UK AI Insurance Analysis")
        print("=" * 60)
        
        # Get all UK articles
        uk_articles = self.db.get_articles(limit=200, region='uk')
        
        # Analyze AI + insurance intersection
        ai_articles = [a for a in uk_articles if a.ai_relevant]
        
        insurance_keywords = ['insurance', 'insurtech', 'underwriting', 'claims', 'risk', 'premium', 'coverage']
        fintech_keywords = ['fintech', 'regtech', 'embedded finance', 'digital banking']
        
        ai_insurance = []
        ai_fintech = []
        
        for article in ai_articles:
            content_text = (article.title + " " + article.content).lower()
            
            if any(kw in content_text for kw in insurance_keywords):
                ai_insurance.append(article)
            
            if any(kw in content_text for kw in fintech_keywords):
                ai_fintech.append(article)
        
        print(f"📊 UK Article Analysis:")
        print(f"   Total UK Articles: {len(uk_articles)}")
        print(f"   AI-Relevant: {len(ai_articles)} ({len(ai_articles)/len(uk_articles)*100:.1f}%)")
        print(f"   AI + Insurance: {len(ai_insurance)} ({len(ai_insurance)/len(ai_articles)*100:.1f}% of AI)")
        print(f"   AI + Fintech: {len(ai_fintech)} ({len(ai_fintech)/len(ai_articles)*100:.1f}% of AI)")
        
        # Keyword frequency analysis
        from collections import Counter
        
        all_ai_keywords = []
        for article in ai_articles:
            if article.ai_keywords_found:
                all_ai_keywords.extend(article.ai_keywords_found)
        
        ai_keyword_counts = Counter(all_ai_keywords)
        
        print(f"\n🔤 Top AI Keywords in UK:")
        for keyword, count in ai_keyword_counts.most_common(10):
            print(f"   {keyword}: {count}")
        
        # Source analysis
        sources = {}
        for article in ai_insurance:
            source = article.source_name
            if source not in sources:
                sources[source] = {'total': 0, 'ai_insurance': 0}
            sources[source]['total'] += 1
            sources[source]['ai_insurance'] += 1
        
        print(f"\n📰 Top Sources for UK AI + Insurance:")
        sorted_sources = sorted(sources.items(), key=lambda x: x[1]['ai_insurance'], reverse=True)
        for source, stats in sorted_sources[:5]:
            if stats['ai_insurance'] > 0:
                print(f"   {source}: {stats['ai_insurance']} articles")
        
        # Recent activity
        recent_ai_insurance = [a for a in ai_insurance if a.published_at and 
                              (datetime.now() - a.published_at.replace(tzinfo=None)).days <= 7]
        
        print(f"\n📅 Recent Activity (Last 7 Days):")
        print(f"   New AI + Insurance articles: {len(recent_ai_insurance)}")
        
        if recent_ai_insurance:
            print(f"   Latest articles:")
            for article in recent_ai_insurance[:3]:
                print(f"     • {article.title[:60]}... ({article.source_name})")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="UK AI Insurance Collection and Search Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect UK AI insurance content
  python uk_ai_insurance_cli.py collect
  
  # Search UK AI insurance content  
  python uk_ai_insurance_cli.py search
  
  # Custom search with details
  python uk_ai_insurance_cli.py search --query "risk assessment AI" --show-details
  
  # Comprehensive analysis
  python uk_ai_insurance_cli.py analyze
  
  # Force collection and search
  python uk_ai_insurance_cli.py collect --force && python uk_ai_insurance_cli.py search
        """
    )
    
    parser.add_argument('command', choices=['collect', 'search', 'analyze'], 
                       help='Command to execute')
    
    # Collection options
    parser.add_argument('--force', action='store_true', 
                       help='Force collection even if recent data exists')
    parser.add_argument('--limit', type=int, 
                       help='Limit articles per feed')
    
    # Search options  
    parser.add_argument('--query', type=str,
                       help='Custom search query (default: comprehensive AI + insurance search)')
    parser.add_argument('--strategy', choices=['smart', 'boolean', 'fuzzy', 'semantic', 'pattern', 'basic'],
                       default='smart', help='Search strategy (default: smart)')
    parser.add_argument('--limit-results', type=int, default=10,
                       help='Limit search results (default: 10)')
    parser.add_argument('--show-details', action='store_true',
                       help='Show detailed search information')
    
    args = parser.parse_args()
    
    # Initialize CLI tool
    cli = UKAIInsuranceCLI()
    
    try:
        if args.command == 'collect':
            cli.collect_uk_ai_insurance(limit=args.limit, force=args.force)
            
        elif args.command == 'search':
            cli.search_uk_ai_insurance(
                query=args.query,
                strategy=args.strategy,
                limit=args.limit_results,
                show_details=args.show_details
            )
            
        elif args.command == 'analyze':
            cli.analyze_uk_ai_insurance()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()