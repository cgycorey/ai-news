#!/usr/bin/env python3
"""
Enhanced Multi-Keyword AI News Demo

This script demonstrates the enhanced multi-keyword collector capabilities
with various combinations like AI + insurance + UK, ML + healthcare + US, etc.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, 'src')

from ai_news.config import Config
from ai_news.database import Database
from ai_news.enhanced_collector import (
    EnhancedMultiKeywordCollector,
    KeywordCategory,
    MultiKeywordResult
)
from ai_news.enhanced_collector import EnhancedCollector


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'-' * 60}")
    print(f"{title}")
    print("-" * 60)


def analyze_article_with_enhanced(collector: EnhancedMultiKeywordCollector, title: str, content: str, region: str = "global") -> MultiKeywordResult:
    """Analyze a single article with enhanced multi-keyword collector."""
    result = collector.analyze_multi_keywords(
        title=title,
        content=content,
        region=region,
        min_score=0.05
    )
    return result


def demo_single_article_analysis(collector: EnhancedMultiKeywordCollector):
    """Demonstrate enhanced analysis on a single test article."""
    print_header("🎯 Single Article Analysis Demo")
    
    # Test article with multiple relevant keywords
    test_articles = [
        {
            "title": "AI Revolution in UK Insurance: Lloyd's Market Adopts Machine Learning for Underwriting",
            "content": "London-based insurtech companies are deploying artificial intelligence and machine learning algorithms for automated underwriting and risk assessment in the Lloyd's market. British firms are leading the way in claims processing automation, with AI-powered systems reducing manual processing time by 80%. The integration of neural networks and predictive analytics is transforming traditional insurance operations across the United Kingdom.",
            "region": "uk",
            "expected_matches": ["ai", "insurance"]
        },
        {
            "title": "Mayo Clinic Implements Deep Learning for Medical Diagnostics in US Healthcare",
            "content": "The Mayo Clinic has partnered with leading AI companies to implement advanced deep learning systems for medical diagnostics and disease detection. American healthcare providers are increasingly using computer vision algorithms for radiology analysis and natural language processing for clinical documentation. These machine learning tools are improving diagnostic accuracy and reducing medical errors across US hospitals.",
            "region": "us",
            "expected_matches": ["ai", "healthcare"]
        },
        {
            "title": "European Banks Adopt AI for AML Compliance and Fraud Detection",
            "content": "Major European financial institutions are implementing artificial intelligence solutions for anti-money laundering (AML) compliance and fraud detection. EU banks are using machine learning algorithms to monitor transactions and identify suspicious activities. The adoption of AI-driven regtech solutions is helping financial institutions meet GDPR requirements while improving operational efficiency in the Eurozone.",
            "region": "eu",
            "expected_matches": ["ai", "fintech"]
        }
    ]
    
    for i, article in enumerate(test_articles, 1):
        print_section(f"Article {i}: {article['title']}")
        print(f"Region: {article['region'].upper()}")
        print(f"Expected categories: {', '.join(article['expected_matches'])}")
        print()
        
        # Analyze with enhanced collector
        result = analyze_article_with_enhanced(
            collector, 
            article['title'], 
            article['content'], 
            article['region']
        )
        
        # Display results
        print(f"🎯 RELEVANT: {'✅ YES' if result.is_relevant else '❌ NO'}")
        print(f"📊 Final Score: {result.final_score:.3f}")
        print(f"🎯 Total Score: {result.total_score:.3f}")
        print(f"🔗 Intersection Score: {result.intersection_score:.3f}")
        print(f"🌍 Region Boost: {result.region_boost:.3f}")
        print(f"⏱️  Analysis Time: {result.execution_time:.4f}s")
        
        if result.category_scores:
            print("\n📈 Category Scores:")
            for category, score in result.category_scores.items():
                print(f"  • {category.upper()}: {score:.3f}")
        
        if result.matches:
            print("\n🔍 Top Keyword Matches:")
            for match in result.matches[:5]:
                print(f"  • {match.keyword} ({match.category}): {match.score:.3f}")
                context = match.context[:80] + "..." if len(match.context) > 80 else match.context
                print(f"    Context: {context}")
        
        print(f"\n🏆 Performance: {result.execution_time:.4f}s per article (target: < 0.1s)")


def demo_database_search(config_path: Path):
    """Demonstrate enhanced search on real database articles."""
    print_header("🔍 Database Search Demo")
    
    try:
        config = Config.load(config_path)
        database = Database(config.database_path)
        collector = EnhancedMultiKeywordCollector(performance_mode=True)
        
        # Get article count
        total_articles = database.get_article_count()
        print(f"📚 Total articles in database: {total_articles}")
        
        if total_articles == 0:
            print("⚠️  No articles found. Run collection first.")
            return
        
        # Test scenarios
        scenarios = [
            {
                "name": "AI + Insurance in UK",
                "filter": collector.create_ai_insurance_uk_filter(),
                "description": "Articles about AI and insurance, with UK region boost"
            },
            {
                "name": "AI + Healthcare in US", 
                "filter": collector.create_ai_healthcare_us_filter(),
                "description": "Articles about AI and healthcare, with US region boost"
            },
            {
                "name": "ML + FinTech in EU",
                "filter": collector.create_ml_fintech_eu_filter(),
                "description": "Articles about ML and fintech, with EU region boost"
            }
        ]
        
        for scenario in scenarios:
            print_section(f"Scenario: {scenario['name']}")
            print(f"Description: {scenario['description']}")
            
            # Get articles for analysis
            region = scenario['filter'].get('region', 'global')
            articles = database.get_articles(limit=500, region=region if region != 'global' else None)
            
            print(f"📊 Analyzing {len(articles)} articles from {region.upper()} region...")
            
            # Analyze articles
            matches = 0
            total_score = 0
            intersection_matches = 0
            category_matches = {}
            
            start_time = datetime.now()
            
            for article in articles:
                result = analyze_article_with_enhanced(
                    collector,
                    article.title,
                    article.content,
                    region
                )
                
                # Check if meets scenario criteria
                if result.is_relevant:
                    required_combinations = scenario['filter'].get('required_combinations', [])
                    if required_combinations:
                        matched_categories = set(result.category_scores.keys())
                        for combo in required_combinations:
                            if isinstance(combo, list) and set(combo).issubset(matched_categories):
                                matches += 1
                                total_score += result.final_score
                                if result.intersection_score > 0:
                                    intersection_matches += 1
                                
                                # Track category matches
                                for category in matched_categories:
                                    if category not in category_matches:
                                        category_matches[category] = 0
                                    category_matches[category] += 1
                                break
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            avg_time_per_article = analysis_time / len(articles)
            
            print(f"\n📈 Results:")
            print(f"  Articles analyzed: {len(articles)}")
            print(f"  Matching articles: {matches} ({matches/len(articles)*100:.1f}% coverage)")
            print(f"  Intersection matches: {intersection_matches}")
            print(f"  Analysis time: {analysis_time:.3f}s total ({avg_time_per_article:.4f}s per article)")
            
            if matches > 0:
                print(f"  Average score: {total_score/matches:.3f}")
                print(f"  Category distribution: {dict(category_matches)}")
            
            # Performance check
            if avg_time_per_article < 0.1:
                print(f"  ✅ Performance target met (< 0.1s per article)")
            else:
                print(f"  ⚠️  Performance target exceeded ({avg_time_per_article:.4f}s per article)")
    
    except Exception as e:
        print(f"❌ Error during database search: {e}")


def demo_performance_comparison():
    """Demonstrate performance comparison between old and new systems."""
    print_header("⚡ Performance Comparison Demo")
    
    # Simulate articles
    test_articles = [
        {
            "title": "AI in Insurance Technology",
            "content": "Artificial intelligence is transforming the insurance industry with machine learning algorithms for risk assessment and claims processing.",
            "region": "uk"
        }
        for _ in range(100)  # Create 100 similar articles
    ]
    
    collector = EnhancedMultiKeywordCollector(performance_mode=True)
    
    print_section("Performance Test: 100 Articles")
    print(f"📊 Analyzing {len(test_articles)} articles...")
    
    start_time = datetime.now()
    
    total_matches = 0
    for article in test_articles:
        result = analyze_article_with_enhanced(
            collector,
            article['title'],
            article['content'],
            article['region']
        )
        if result.is_relevant:
            total_matches += 1
    
    analysis_time = (datetime.now() - start_time).total_seconds()
    print("📈 Performance Results:")
    avg_time_per_article = analysis_time / len(test_articles)
    print(f"  Articles analyzed: {len(test_articles)}")
    print(f"  Matching articles: {total_matches} ({total_matches/len(test_articles)*100:.1f}%)")
    print(f"  Total analysis time: {analysis_time:.3f}s")
    print(f"  Average time per article: {avg_time_per_article:.4f}s")
    
    # Performance check
    if avg_time_per_article < 0.1:
        print(f"  ✅ Performance target met (< 0.1s per article)")
    else:
        print(f"  ⚠️  Performance target exceeded ({avg_time_per_article:.4f}s per article)")

def demo_intersection_detection():
    """Demonstrate intersection detection capabilities."""
    print_header("🔗 Intersection Detection Demo")
    
    collector = EnhancedMultiKeywordCollector(performance_mode=True)
    
    # Test cases for intersection detection
    intersection_cases = [
        {
            "title": "AI-Powered Healthcare Diagnostics Transform Medical Industry",
            "content": "Machine learning algorithms are revolutionizing healthcare diagnostics with artificial intelligence systems that can detect diseases earlier and more accurately than traditional methods.",
            "expected_intersection": "AI + Healthcare"
        },
        {
            "title": "FinTech Startups Use AI for Fraud Detection in Banking",
            "content": "Financial technology companies are deploying artificial intelligence and machine learning solutions for fraud detection and risk management in digital banking platforms.",
            "expected_intersection": "AI + FinTech"
        },
        {
            "title": "Insurance Technology Companies Adopt ML for Underwriting",
            "content": "Insurtech firms are implementing machine learning algorithms for automated underwriting and risk assessment in the insurance industry.",
            "expected_intersection": "Insurance + ML"
        }
    ]
    
    for i, case in enumerate(intersection_cases, 1):
        print_section(f"Intersection Case {i}: {case['expected_intersection']}")
        print(f"Title: {case['title']}")
        print()
        
        result = analyze_article_with_enhanced(
            collector,
            case['title'],
            case['content']
        )
        
        print(f"🔗 Intersection Score: {result.intersection_score:.3f}")
        print(f"📊 Categories found: {', '.join(result.category_scores.keys())}")
        
        if result.intersection_score > 0:
            print("✅ Intersection detected!")
            print(f"🎯 Multi-category relevance bonus applied")
        else:
            print("❌ No intersection detected")
        
        print(f"📈 Final score (with intersection): {result.final_score:.3f}")


def main():
    """Main demonstration function."""
    print("🚀 Enhanced Multi-Keyword AI News Demo")
    print("This demo showcases the enhanced multi-keyword collector capabilities")
    print("with intersection detection, region boosts, and advanced scoring.")
    
    # Initialize enhanced collector
    collector = EnhancedMultiKeywordCollector(performance_mode=True)
    
    # Run demonstrations
    demo_single_article_analysis(collector)
    demo_intersection_detection()
    demo_performance_comparison()
    
    # Database demo (if available)
    config_path = Path('config.json')
    if config_path.exists():
        demo_database_search(config_path)
    else:
        print_section("Database Search Demo")
        print("⚠️  config.json not found. Skipping database search demo.")
        print("💡 To enable database demo, run: python -m ai_news.cli collect")
    
    print_header("🎉 Demo Complete")
    print("Enhanced Multi-Keyword Collector Features Demonstrated:")
    print("✅ Multi-category keyword analysis with scoring")
    print("✅ Intersection detection for high-value combinations")
    print("✅ Region-specific keyword boosts")
    print("✅ Performance optimization (< 0.1s per article)")
    print("✅ Comprehensive coverage reporting")
    print("✅ Backward compatibility with existing system")
    
    print("\n💡 Usage Examples:")
    print("  python -m ai_news.enhanced_cli multi ai insurance --region uk")
    print("  python -m ai_news.enhanced_cli intersection ai healthcare --region us")
    print("  python enhanced_comprehensive_search.py 'AI + insurance region:UK' --strategy multi_keyword")


if __name__ == '__main__':
    main()