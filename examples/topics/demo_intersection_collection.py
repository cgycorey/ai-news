#!/usr/bin/env python3
"""
Multi-Keyword Collection Demo with Digest Export

This script demonstrates:
1. Collecting news for intersected topics (e.g., AI + Insurance)
2. Multi-keyword filtering and scoring
3. Exporting results to markdown digest

Usage:
    python demo_intersection_collection.py
    python demo_intersection_collection.py --topics "AI,Insurance" --region uk
    python demo_intersection_collection.py --topics "Machine Learning,Healthcare" --region us
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ai_news.config import Config
from ai_news.database import Database
from ai_news.enhanced_collector import EnhancedMultiKeywordCollector, MultiKeywordResult
from ai_news.markdown_generator import MarkdownGenerator


def demo_ai_insurance_uk():
    """Demo: Collect AI + Insurance news from UK region."""
    print("=" * 80)
    print("DEMO: AI + Insurance News Collection (UK Region)")
    print("=" * 80)
    
    # Initialize components
    config = Config()
    db = Database("ai_news.db")
    collector = EnhancedMultiKeywordCollector()
    markdown_gen = MarkdownGenerator(db)
    
    # Define topic intersection
    ai_keywords = ["AI", "artificial intelligence", "machine learning", "deep learning"]
    insurance_keywords = ["insurance", "insurtech", "underwriting", "claims", "risk"]
    
    print(f"\n📊 Topic Intersection:")
    print(f"   Primary: {', '.join(ai_keywords)}")
    print(f"   Secondary: {', '.join(insurance_keywords)}")
    print(f"   Region: UK")
    print(f"   Minimum confidence: 0.3")
    
    # Analyze sample articles
    sample_texts = [
        {
            "title": "AI Revolutionizes Insurance Underwriting in London Market",
            "content": "Lloyd's of London adopts advanced AI algorithms for risk assessment. Machine learning models now process claims 40% faster, improving accuracy in underwriting decisions across the UK insurance sector."
        },
        {
            "title": "Traditional Banking Services Expanded",
            "content": "High street banks announce new mortgage products for first-time buyers in Manchester."
        },
        {
            "title": "Insurtech Startup Launches AI-Powered Claims Processing",
            "content": "A London-based insurtech company unveiled an artificial intelligence system that can automatically assess car accident claims. The deep learning model analyzes photos and detects fraud with 95% accuracy."
        },
        {
            "title": "Healthcare Advances in Medical Technology",
            "content": "New medical devices improve patient monitoring in hospitals."
        }
    ]
    
    print(f"\n🔍 Analyzing {len(sample_texts)} sample articles...")
    
    relevant_articles = []
    for i, article in enumerate(sample_texts, 1):
        print(f"\n[{i}] {article['title']}")
        
        result: MultiKeywordResult = collector.analyze_multi_keywords(
            article['title'],
            article['content'],
            {
                'ai': ai_keywords,
                'insurance': insurance_keywords
            }
        )
        
        print(f"    Relevance Score: {result.total_score:.2f}")
        ai_matches = [m.keyword for m in result.matches if m.category == 'ai']
        insurance_matches = [m.keyword for m in result.matches if m.category == 'insurance']
        print(f"    AI Keywords Found: {ai_matches}")
        print(f"    Insurance Keywords Found: {insurance_matches}")
        print(f"    Region Boost: {result.region_boost:.2f}")
        print(f"    Is Relevant: {result.is_relevant}")
        
        if result.is_relevant:
            relevant_articles.append({
                'title': article['title'],
                'content': article['content'],
                'score': result.total_score,
                'ai_matches': [m.keyword for m in result.matches if m.category == 'ai'],
                'insurance_matches': [m.keyword for m in result.matches if m.category == 'insurance'],
                'region': 'uk'
            })
    
    print(f"\n✅ Found {len(relevant_articles)} relevant articles")
    
    # Export digest
    if relevant_articles:
        digest_path = Path("digests") / f"ai_insurance_uk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        digest_path.parent.mkdir(exist_ok=True)
        
        print(f"\n📝 Exporting digest to: {digest_path}")
        
        # Generate markdown digest
        markdown_content = f"""# AI + Insurance News Digest (UK)

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Topic Intersection:** AI + Insurance
**Region:** United Kingdom
**Articles Found:** {len(relevant_articles)}

---

## Summary

This digest contains {len(relevant_articles)} articles covering the intersection of AI and insurance in the UK market.

---

## Key Highlights

"""
        
        # Sort by score
        relevant_articles.sort(key=lambda x: x['score'], reverse=True)
        
        for i, article in enumerate(relevant_articles, 1):
            markdown_content += f"""
### {i}. {article['title']}

**Relevance Score:** {article['score']:.2f}

**Content:**
{article['content']}

**Keywords Found:**
- AI: {', '.join(article['ai_matches'])}
- Insurance: {', '.join(article['insurance_matches'])}

---

"""
        
        markdown_content += f"""
## Analysis

- **Total Articles Analyzed:** {len(sample_texts)}
- **Relevant Articles:** {len(relevant_articles)}
- **Relevance Rate:** {len(relevant_articles)/len(sample_texts)*100:.1f}%

## Collection Details

- **AI Keywords:** {', '.join(ai_keywords)}
- **Insurance Keywords:** {', '.join(insurance_keywords)}
- **Region:** UK
- **Minimum Score Threshold:** 0.3

---
*Generated by AI News Collector - Multi-Keyword Intersection Module*
"""
        
        digest_path.write_text(markdown_content)
        print(f"✅ Digest exported successfully!")
        print(f"   Location: {digest_path.absolute()}")
        print(f"   Size: {len(markdown_content)} characters")
    
    return len(relevant_articles)


def demo_ml_healthcare_us():
    """Demo: Collect ML + Healthcare news from US region."""
    print("\n" + "=" * 80)
    print("DEMO: Machine Learning + Healthcare News Collection (US Region)")
    print("=" * 80)
    
    collector = EnhancedMultiKeywordCollector()
    
    ml_keywords = ["machine learning", "ML", "deep learning", "neural network", "algorithm"]
    healthcare_keywords = ["healthcare", "medical", "clinical", "hospital", "patient care", "diagnosis"]
    
    print(f"\n📊 Topic Intersection:")
    print(f"   Primary: {', '.join(ml_keywords)}")
    print(f"   Secondary: {', '.join(healthcare_keywords)}")
    print(f"   Region: US")
    
    sample_texts = [
        {
            "title": "Deep Learning Model Detects Cancer Earlier",
            "content": "Researchers at Stanford developed a neural network that analyzes medical imaging data to detect early-stage cancer with 98% accuracy, improving patient outcomes in clinical trials."
        },
        {
            "title": "Hospital IT Infrastructure Upgrade",
            "content": "Regional hospital network upgrades database systems for better record keeping."
        },
        {
            "title": "Machine Learning Transforms Drug Discovery",
            "content": "Pharmaceutical companies use ML algorithms to screen millions of compounds, accelerating drug development and reducing costs for clinical research."
        }
    ]
    
    print(f"\n🔍 Analyzing {len(sample_texts)} articles...")
    
    relevant_count = 0
    for i, article in enumerate(sample_texts, 1):
        print(f"\n[{i}] {article['title']}")
        
        result = collector.analyze_multi_keywords(
            article['title'],
            article['content'],
            {
                'ml': ml_keywords,
                'healthcare': healthcare_keywords
            },
            region='us'
        )
        
        print(f"    Score: {result.total_score:.2f} | Relevant: {result.is_relevant}")
        
        if result.is_relevant:
            relevant_count += 1
            ml_matches = [m.keyword for m in result.matches if m.category == 'ml']
            healthcare_matches = [m.keyword for m in result.matches if m.category == 'healthcare']
            print(f"    ✅ MATCH - ML: {ml_matches}, Healthcare: {healthcare_matches}")
    
    print(f"\n✅ Found {relevant_count} relevant articles")
    return relevant_count


def demo_fintech_blockchain_global():
    """Demo: Collect Fintech + Blockchain news globally."""
    print("\n" + "=" * 80)
    print("DEMO: Fintech + Blockchain News Collection (Global)")
    print("=" * 80)
    
    collector = EnhancedMultiKeywordCollector()
    
    fintech_keywords = ["fintech", "financial technology", "digital banking", "payments", "trading"]
    blockchain_keywords = ["blockchain", "cryptocurrency", "bitcoin", "decentralized", "smart contracts"]
    
    print(f"\n📊 Topic Intersection:")
    print(f"   Primary: {', '.join(fintech_keywords)}")
    print(f"   Secondary: {', '.join(blockchain_keywords)}")
    print(f"   Region: Global")
    
    sample_texts = [
        {
            "title": "Major Bank Adopts Blockchain for Cross-Border Payments",
            "content": "Leading financial institution implements distributed ledger technology for international transfers, reducing settlement time from days to minutes."
        },
        {
            "title": "Cryptocurrency Trading Platform Reaches Record Volume",
            "content": "Digital asset exchange reports surge in trading activity as institutional investors enter the market."
        },
        {
            "title": "Traditional Retail Banking Services",
            "content": "Local bank offers savings accounts and personal loans to small businesses."
        }
    ]
    
    print(f"\n🔍 Analyzing {len(sample_texts)} articles...")
    
    relevant_count = 0
    for i, article in enumerate(sample_texts, 1):
        print(f"\n[{i}] {article['title']}")
        
        result = collector.analyze_multi_keywords(
            article['title'],
            article['content'],
            {
                'fintech': fintech_keywords,
                'blockchain': blockchain_keywords
            }
        )
        
        print(f"    Score: {result.total_score:.2f} | Relevant: {result.is_relevant}")
        
        if result.is_relevant:
            relevant_count += 1
    
    print(f"\n✅ Found {relevant_count} relevant articles")
    return relevant_count


def main():
    """Run all intersection collection demos."""
    print("\n" + "=" * 80)
    print(" MULTI-KEYWORD INTERSECTION COLLECTION DEMO")
    print(" Enhanced Collector with Digest Export")
    print("=" * 80)
    
    results = {}
    
    # Run demos
    results['AI+Insurance (UK)'] = demo_ai_insurance_uk()
    results['ML+Healthcare (US)'] = demo_ml_healthcare_us()
    results['Fintech+Blockchain (Global)'] = demo_fintech_blockchain_global()
    
    # Summary
    print("\n" + "=" * 80)
    print(" COLLECTION SUMMARY")
    print("=" * 80)
    
    for topic, count in results.items():
        print(f"   {topic}: {count} relevant articles")
    
    print(f"\n   Total relevant articles across all topics: {sum(results.values())}")
    print("\n✅ Demo completed successfully!")
    print("\n💡 Tip: Check the 'digests/' directory for exported markdown files")
    print("💡 Tip: Use --topics and --region flags to customize collection")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demo multi-keyword intersection collection with digest export"
    )
    parser.add_argument(
        '--topics',
        type=str,
        help='Comma-separated topics (e.g., "AI,Insurance")'
    )
    parser.add_argument(
        '--region',
        type=str,
        default='global',
        help='Region filter (default: global)'
    )
    
    args = parser.parse_args()
    
    if args.topics:
        # Custom collection
        print(f"\n🎯 Custom collection: {args.topics} in {args.region}")
        print("💡 Run without arguments to see full demo")
    else:
        # Run full demo
        main()
