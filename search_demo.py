#!/usr/bin/env python3
"""
Simple demonstration of flexible search strategies for AI News.
"""

import sys
import re
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from src.ai_news.config import Config
from src.ai_news.database import Database

def demonstrate_flexible_searches(database, limit=10):
    """Demonstrate various flexible search strategies."""
    
    # Get sample articles
    articles = database.get_articles(limit=50, ai_only=False)
    
    print("🔍 Flexible Search Strategies Demonstration")
    print("=" * 60)
    
    # 1. Pattern/Regex Search
    print("\n1. 🎯 REGEX SEARCH: Articles containing 'AI' or 'artificial'")
    ai_pattern = re.compile(r'\b(artificial\s+intelligence|AI)\b', re.IGNORECASE)
    regex_results = []
    
    for article in articles:
        text = f"{article.title} {article.content} {article.summary}".lower()
        matches = ai_pattern.findall(text)
        if matches:
            regex_results.append((article, len(matches)))
    
    regex_results.sort(key=lambda x: x[1], reverse=True)
    for article, count in regex_results[:5]:
        print(f"   📄 {article.title[:60]}... ({count} matches)")
    
    # 2. Field-Specific Search  
    print("\n2. 📋 TITLE-ONLY SEARCH: Articles with 'insurance' in title")
    title_results = []
    for article in articles:
        if 'insurance' in article.title.lower():
            title_results.append(article)
    
    for article in title_results[:5]:
        print(f"   📄 {article.title}")
        print(f"      Source: {article.source_name}")
    
    # 3. Combined Criteria Search
    print("\n3. 🎨 COMBINED SEARCH: AI-relevant UK articles")
    combined_results = []
    for article in articles:
        # Check AI relevance (simplified)
        ai_keywords = ['ai', 'artificial', 'machine learning', 'intelligence', 'chatgpt']
        text_lower = f"{article.title} {article.content}".lower()
        is_ai_relevant = any(keyword in text_lower for keyword in ai_keywords)
        
        # Check if UK region (simplified)
        is_uk = ('uk' in article.source_name.lower() or 
                'british' in text_lower or 
                'london' in text_lower)
        
        if is_ai_relevant and is_uk:
            combined_results.append(article)
    
    for article in combined_results[:3]:
        print(f"   📄 {article.title}")
        print(f"      Source: {article.source_name} | AI: {article.ai_relevant}")
    
    # 4. Proximity Search Simulation
    print("\n4. 📏 PROXIMITY SEARCH: 'AI' near 'insurance' within 50 words")
    proximity_results = []
    max_distance = 50
    
    for article in articles:
        words = f"{article.title} {article.content}".split()
        
        # Find positions of keywords
        ai_positions = [i for i, word in enumerate(words) 
                       if 'ai' in word.lower() or 'artificial' in word.lower()]
        insurance_positions = [i for i, word in enumerate(words) 
                             if 'insurance' in word.lower()]
        
        # Check proximity
        for ai_pos in ai_positions:
            for ins_pos in insurance_positions:
                if abs(ai_pos - ins_pos) <= max_distance:
                    proximity_results.append((article, abs(ai_pos - ins_pos)))
                    break
    
    # Sort by closest proximity
    proximity_results.sort(key=lambda x: x[1])
    for article, distance in proximity_results[:3]:
        print(f"   📄 {article.title[:60]}... (distance: {distance} words)")
    
    # 5. Source-Based Search
    print("\n5. 📰 SOURCE-BASED SEARCH: Articles from specific sources")
    target_sources = ['bbc', 'techcrunch', 'venturebeat']
    source_results = {}
    
    for article in articles:
        for source in target_sources:
            if source in article.source_name.lower():
                if source not in source_results:
                    source_results[source] = []
                source_results[source].append(article)
                break
    
    for source, articles_list in source_results.items():
        print(f"   📰 {source.upper()}: {len(articles_list)} articles")
        for article in articles_list[:2]:
            print(f"      - {article.title[:50]}...")
    
    # 6. Category-Based Search
    print("\n6. 🏷️  CATEGORY SEARCH: Articles by category")
    category_counts = {}
    category_examples = {}
    
    for article in articles:
        category = article.category
        if category not in category_counts:
            category_counts[category] = 0
            category_examples[category] = []
        category_counts[category] += 1
        if len(category_examples[category]) < 3:
            category_examples[category].append(article.title[:50] + "...")
    
    # Sort categories by count
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    
    for category, count in sorted_categories[:8]:
        print(f"   🏷️  {category}: {count} articles")
        for example in category_examples[category][:2]:
            print(f"      - {example}")
    
    # 7. Date-Based Search
    print("\n7. 📅 DATE-BASED SEARCH: Most recent articles")
    recent_articles = [a for a in articles if a.published_at]
    
    # Sort safely, handling both aware and naive datetimes
    def get_sort_key(article):
        if article.published_at:
            # Convert to naive datetime for comparison if it's aware
            if hasattr(article.published_at, 'tzinfo') and article.published_at.tzinfo is not None:
                return article.published_at.replace(tzinfo=None)
            return article.published_at
        return datetime.min
    
    recent_articles.sort(key=get_sort_key, reverse=True)
    
    for article in recent_articles[:5]:
        date_str = article.published_at.strftime("%Y-%m-%d %H:%M") if article.published_at else "Unknown"
        print(f"   📅 {date_str}: {article.title[:60]}...")
    
    print(f"\n📊 SEARCH SUMMARY:")
    print(f"   Total articles analyzed: {len(articles)}")
    print(f"   Regex matches found: {len(regex_results)}")
    print(f"   Title matches found: {len(title_results)}")
    print(f"   AI+UK combined: {len(combined_results)}")
    print(f"   Proximity matches: {len(proximity_results)}")
    print(f"   Sources found: {len(source_results)}")
    print(f"   Categories found: {len(category_counts)}")
    print(f"   Recent articles: {len(recent_articles)}")

def simple_search_examples(database):
    """Show simple but effective search combinations."""
    
    print("\n\n🔧 SIMPLE SEARCH COMBINATIONS")
    print("=" * 40)
    
    # Get all articles
    all_articles = database.get_articles(limit=100)
    
    # Example 1: Multi-keyword OR logic
    print("\n1️⃣ OR Logic: Articles containing 'AI' OR 'insurance' OR 'fintech'")
    or_keywords = ['ai', 'insurance', 'fintech', 'technology']
    or_results = []
    
    for article in all_articles:
        text = f"{article.title} {article.content} {article.summary}".lower()
        if any(keyword in text for keyword in or_keywords):
            found_keywords = [kw for kw in or_keywords if kw in text]
            or_results.append((article, found_keywords))
    
    print(f"   Found {len(or_results)} articles")
    for article, keywords in or_results[:5]:
        print(f"   📄 {article.title[:50]}... (Keywords: {', '.join(keywords)})")
    
    # Example 2: Multi-keyword AND logic
    print("\n2️⃣ AND Logic: Articles containing both 'AI' AND 'insurance'")
    and_keywords = ['ai', 'insurance']
    and_results = []
    
    for article in all_articles:
        text = f"{article.title} {article.content} {article.summary}".lower()
        if all(keyword in text for keyword in and_keywords):
            and_results.append(article)
    
    print(f"   Found {len(and_results)} articles")
    for article in and_results[:3]:
        print(f"   📄 {article.title}")
    
    # Example 3: Exclusion logic (NOT)
    print("\n3️⃣ NOT Logic: Articles about technology but NOT 'sports'")
    include_keyword = 'technology'
    exclude_keyword = 'sport'
    exclude_results = []
    
    for article in all_articles:
        text = f"{article.title} {article.content} {article.summary}".lower()
        if include_keyword in text and exclude_keyword not in text:
            exclude_results.append(article)
    
    print(f"   Found {len(exclude_results)} articles")
    for article in exclude_results[:3]:
        print(f"   📄 {article.title}")
    
    # Example 4: Fuzzy matching
    print("\n4️⃣ FUZZY MATCH: Articles with variations of 'company' (company, companies, corp)")
    fuzzy_patterns = ['company', 'companies', 'corp', 'corporation', 'business']
    fuzzy_results = []
    
    for article in all_articles:
        text = f"{article.title} {article.content} {article.summary}".lower()
        found_patterns = [pattern for pattern in fuzzy_patterns if pattern in text]
        if found_patterns:
            fuzzy_results.append((article, found_patterns))
    
    print(f"   Found {len(fuzzy_results)} articles")
    for article, patterns in fuzzy_results[:3]:
        print(f"   📄 {article.title[:50]}... (Patterns: {', '.join(patterns)})")
    
    # Example 5: Regional + Keyword combination
    print("\n5️⃣ REGION + KEYWORD: UK articles with 'government' or 'policy'")
    region_keyword_results = []
    
    for article in all_articles:
        text = f"{article.title} {article.content} {article.summary}".lower()
        is_uk = ('uk' in article.source_name.lower() or 
                'british' in text.lower() or 
                'london' in text.lower())
        
        has_keyword = any(kw in text for kw in ['government', 'policy', 'parliament', 'minister'])
        
        if is_uk and has_keyword:
            region_keyword_results.append(article)
    
    print(f"   Found {len(region_keyword_results)} UK articles with government keywords")
    for article in region_keyword_results[:3]:
        print(f"   📄 {article.title}")
        print(f"      Source: {article.source_name}")

def advanced_search_techniques(database):
    """Show advanced search techniques and suggestions."""
    
    print("\n\n🚀 ADVANCED SEARCH TECHNIQUES")
    print("=" * 40)
    
    # Get sample data
    all_articles = database.get_articles(limit=100)
    
    print("\n💡 SEARCH STRATEGY SUGGESTIONS:")
    
    print("\n1. 🎯 TARGETED KEYWORD SEARCH:")
    print("   Use specific industry terms instead of generic ones")
    print("   ❌ 'technology' → ✅ 'insurtech', 'fintech', 'healthtech'")
    print("   ❌ 'news' → ✅ 'regulation', 'compliance', 'acquisition'")
    
    print("\n2. 🎨 BOOLEAN LOGIC EXAMPLES:")
    print("   AI AND (insurance OR fintech) -> Articles about AI in finance")
    print("   (london OR uk) AND startup -> UK startup articles")
    print("   NOT (sports OR entertainment) -> Exclude entertainment content")
    
    print("\n3. 📊 FILTER COMBINATIONS:")
    print("   • Region + AI + Time Range: UK AI articles from last week")
    print("   • Source + Category: TechCrunch startup articles")
    print("   • Relevance + Keyword: High-AI relevance articles about 'chatbot'")
    
    print("\n4. 🔍 PRACTICAL SEARCH EXAMPLES:")
    
    # Demonstrate some practical combinations
    print("\n   🏢 BUSINESS + AI:")
    business_ai = []
    for article in all_articles:
        text = f"{article.title} {article.content}".lower()
        if (any(term in text for term in ['business', 'company', 'corporation']) and
            any(ai_term in text for ai_term in ['ai', 'artificial', 'machine learning'])):
            business_ai.append(article)
    
    print(f"      Found {len(business_ai)} articles")
    for article in business_ai[:3]:
        print(f"      - {article.title[:60]}...")
    
    print("\n   🏥 HEALTH + TECHNOLOGY:")
    health_tech = []
    for article in all_articles:
        text = f"{article.title} {article.content}".lower()
        if (any(term in text for term in ['health', 'medical', 'healthcare']) and
            any(tech_term in text for tech_term in ['technology', 'digital', 'software'])):
            health_tech.append(article)
    
    print(f"      Found {len(health_tech)} articles")
    for article in health_tech[:3]:
        print(f"      - {article.title[:60]}...")
    
    print("\n   📈 INVESTMENT + STARTUP:")
    investment_startup = []
    for article in all_articles:
        text = f"{article.title} {article.content}".lower()
        if (any(term in text for term in ['investment', 'funding', 'venture', 'capital']) and
            any(startup_term in text for startup_term in ['startup', 'entrepreneur', 'scale-up'])):
            investment_startup.append(article)
    
    print(f"      Found {len(investment_startup)} articles")
    for article in investment_startup[:3]:
        print(f"      - {article.title[:60]}...")

def main():
    parser = argparse.ArgumentParser(description='Flexible search demonstrations for AI News')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--limit', type=int, default=50, help='Number of articles to analyze')
    
    args = parser.parse_args()
    
    # Load configuration and database
    config_path = Path(args.config)
    config = Config.load(config_path)
    database = Database(config.database_path)
    
    # Run demonstrations
    demonstrate_flexible_searches(database, args.limit)
    simple_search_examples(database)
    advanced_search_techniques(database)
    
    print(f"\n\n🎯 QUICK SEARCH TIPS:")
    print(f"• Use specific industry terminology")
    print(f"• Combine regional filters with keywords")
    print(f"• Leverage AI-relevance scoring")
    print(f"• Use topic digests for comprehensive coverage")
    print(f"• Schedule regular collection for current data")

if __name__ == '__main__':
    main()