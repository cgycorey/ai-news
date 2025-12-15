#!/usr/bin/env python3
"""
Performance Comparison: Original vs Enhanced Multi-Keyword Collection

This script compares the original SimpleCollector with the enhanced multi-keyword system
to demonstrate improvements in accuracy, scoring, and functionality.
"""

import sys
import os
import time
from typing import Dict, List, Any, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_news.collector import SimpleCollector
from ai_news.database import Database, Article
from enhanced_multi_keyword_collector import EnhancedMultiKeywordCollector


def create_test_articles() -> List[Dict[str, Any]]:
    """Create comprehensive test articles for comparison."""
    return [
        {
            "title": "AI Revolution in UK Insurance Underwriting",
            "content": "London-based insurtech companies are deploying artificial intelligence and machine learning algorithms for automated underwriting and risk assessment in the Lloyd's market.",
            "region": "uk",
            "expected_categories": ["ai", "insurance"],
            "description": "Perfect AI + insurance match with UK context"
        },
        {
            "title": "Machine Learning in Medical Diagnostics",
            "content": "Deep learning models are revolutionizing healthcare diagnostics with AI-powered imaging analysis that can detect diseases earlier than traditional methods.",
            "region": "us",
            "expected_categories": ["ai", "healthcare"],
            "description": "AI + healthcare match with technical depth"
        },
        {
            "title": "European Banks Adopt ML for Fraud Detection",
            "content": "EU financial institutions implement machine learning algorithms for real-time fraud detection and anti-money laundering compliance in banking systems.",
            "region": "eu",
            "expected_categories": ["ai", "fintech"],
            "description": "ML + fintech match with compliance focus"
        },
        {
            "title": "AI-Powered Platform for Healthcare Insurance Claims",
            "content": "Artificial intelligence system processes healthcare insurance claims using machine learning for medical record analysis and fraud detection.",
            "region": "global",
            "expected_categories": ["ai", "healthcare", "insurance"],
            "description": "Complex multi-category intersection"
        },
        {
            "title": "Local Sports Team Wins Championship",
            "content": "The local basketball team won the championship game in an exciting match against their rivals from the neighboring city.",
            "region": "global",
            "expected_categories": [],
            "description": "Negative control - should not match"
        },
        {
            "title": "Tech Company Announces New Software Update",
            "content": "The software company released a new version of their application with improved user interface and bug fixes.",
            "region": "global",
            "expected_categories": [],
            "description": "Another negative control"
        }
    ]


def test_original_collector(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Test the original SimpleCollector."""
    print("🔍 Testing Original SimpleCollector")
    print("=" * 60)
    
    # Initialize original collector
    db = Database(":memory:")  # In-memory database for testing
    collector = SimpleCollector(db)
    
    # Comprehensive keyword list (simulating combined categories)
    comprehensive_keywords = [
        # AI keywords
        "AI", "artificial intelligence", "machine learning", "deep learning",
        "LLM", "GPT", "ChatGPT", "neural network", "algorithm", "automation",
        # Insurance keywords
        "insurance", "insurtech", "underwriting", "claims", "risk", "premium", "coverage",
        # Healthcare keywords
        "healthcare", "medical", "diagnostics", "medicine", "clinical", "hospital",
        # Fintech keywords
        "fintech", "banking", "financial", "trading", "payments", "fraud detection"
    ]
    
    results = {
        "total_articles": len(articles),
        "ai_relevant": 0,
        "execution_times": [],
        "detected_keywords": [],
        "accuracy_results": []
    }
    
    for i, article in enumerate(articles, 1):
        print(f"\n{i}. {article['title'][:50]}...")
        print(f"   Expected: {article['expected_categories']}")
        
        start_time = time.time()
        is_relevant, found_keywords = collector.is_ai_relevant(
            article["title"], article["content"], comprehensive_keywords
        )
        execution_time = time.time() - start_time
        
        results["execution_times"].append(execution_time)
        results["detected_keywords"].extend(found_keywords)
        
        if is_relevant:
            results["ai_relevant"] += 1
        
        # Check accuracy
        expected_ai = bool(article["expected_categories"])
        is_correct = (expected_ai == is_relevant)
        
        print(f"   Actual AI: {is_relevant}")
        print(f"   Keywords: {found_keywords[:5]}...")  # Show first 5
        print(f"   Time: {execution_time:.4f}s")
        print(f"   ✅ Correct" if is_correct else f"   ❌ Incorrect")
        
        results["accuracy_results"].append(is_correct)
    
    # Calculate metrics
    results["avg_execution_time"] = sum(results["execution_times"]) / len(results["execution_times"])
    results["accuracy"] = sum(results["accuracy_results"]) / len(results["accuracy_results"])
    results["ai_relevance_rate"] = results["ai_relevant"] / results["total_articles"]
    
    print(f"\n📊 Original Collector Summary:")
    print(f"   Total articles: {results['total_articles']}")
    print(f"   AI-relevant: {results['ai_relevant']} ({results['ai_relevance_rate']*100:.1f}%)")
    print(f"   Accuracy: {results['accuracy']*100:.1f}%")
    print(f"   Avg execution time: {results['avg_execution_time']:.4f}s")
    
    return results


def test_enhanced_collector(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Test the enhanced multi-keyword collector."""
    print("\n🚀 Testing Enhanced Multi-Keyword Collector")
    print("=" * 60)
    
    collector = EnhancedMultiKeywordCollector(performance_mode=True)
    
    results = {
        "total_articles": len(articles),
        "relevant_articles": 0,
        "execution_times": [],
        "category_matches": {},
        "intersection_matches": 0,
        "accuracy_results": [],
        "score_distribution": [0, 0, 0, 0],  # Score buckets
        "detailed_results": []
    }
    
    for i, article in enumerate(articles, 1):
        print(f"\n{i}. {article['title'][:50]}...")
        print(f"   Expected: {article['expected_categories']}")
        
        start_time = time.time()
        analysis_result = collector.analyze_multi_keywords(
            title=article["title"],
            content=article["content"],
            region=article["region"],
            min_score=0.1
        )
        execution_time = time.time() - start_time
        
        results["execution_times"].append(execution_time)
        
        if analysis_result.is_relevant:
            results["relevant_articles"] += 1
        
        # Track category matches
        for category in analysis_result.category_scores.keys():
            if category not in results["category_matches"]:
                results["category_matches"][category] = 0
            results["category_matches"][category] += 1
        
        # Track intersections
        if len(analysis_result.category_scores) >= 2:
            results["intersection_matches"] += 1
        
        # Score distribution
        score = analysis_result.final_score
        if score <= 0.25:
            results["score_distribution"][0] += 1
        elif score <= 0.5:
            results["score_distribution"][1] += 1
        elif score <= 0.75:
            results["score_distribution"][2] += 1
        else:
            results["score_distribution"][3] += 1
        
        # Check accuracy (categories detected vs expected)
        detected_categories = set(analysis_result.category_scores.keys())
        expected_categories = set(article["expected_categories"])
        
        # Consider it correct if:
        # 1. No categories expected and none detected, OR
        # 2. At least one expected category detected
        if not expected_categories:
            is_correct = len(detected_categories) == 0
        else:
            is_correct = bool(detected_categories & expected_categories)
        
        print(f"   Actual categories: {list(detected_categories)}")
        print(f"   Final score: {analysis_result.final_score:.3f}")
        print(f"   Intersection score: {analysis_result.intersection_score:.3f}")
        print(f"   Region boost: {analysis_result.region_boost:.3f}")
        print(f"   Top keywords: {[m.keyword for m in analysis_result.matches[:3]]}")
        print(f"   Time: {execution_time:.4f}s")
        print(f"   ✅ Correct" if is_correct else f"   ❌ Incorrect")
        
        results["accuracy_results"].append(is_correct)
        results["detailed_results"].append({
            "title": article["title"],
            "expected": article["expected_categories"],
            "detected": list(detected_categories),
            "score": analysis_result.final_score,
            "correct": is_correct
        })
    
    # Calculate metrics
    results["avg_execution_time"] = sum(results["execution_times"]) / len(results["execution_times"])
    results["accuracy"] = sum(results["accuracy_results"]) / len(results["accuracy_results"])
    results["relevance_rate"] = results["relevant_articles"] / results["total_articles"]
    results["intersection_rate"] = results["intersection_matches"] / results["total_articles"]
    
    print(f"\n📊 Enhanced Collector Summary:")
    print(f"   Total articles: {results['total_articles']}")
    print(f"   Relevant: {results['relevant_articles']} ({results['relevance_rate']*100:.1f}%)")
    print(f"   Accuracy: {results['accuracy']*100:.1f}%")
    print(f"   Intersection rate: {results['intersection_rate']*100:.1f}%")
    print(f"   Categories detected: {list(results['category_matches'].keys())}")
    print(f"   Avg execution time: {results['avg_execution_time']:.4f}s")
    
    return results


def compare_collectors(original_results: Dict[str, Any], enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare results between original and enhanced collectors."""
    print("\n📊 Comparative Analysis")
    print("=" * 80)
    
    comparison = {
        "accuracy_improvement": enhanced_results["accuracy"] - original_results["accuracy"],
        "execution_time_change": enhanced_results["avg_execution_time"] - original_results["avg_execution_time"],
        "original_accuracy": original_results["accuracy"],
        "enhanced_accuracy": enhanced_results["accuracy"],
        "original_time": original_results["avg_execution_time"],
        "enhanced_time": enhanced_results["avg_execution_time"],
        "enhanced_features": []
    }
    
    print(f"🎯 Key Performance Metrics:")
    print(f"   Accuracy: {original_results['accuracy']*100:.1f}% → {enhanced_results['accuracy']*100:.1f}% ({comparison['accuracy_improvement']*100:+.1f}%)")
    print(f"   Execution Time: {original_results['avg_execution_time']:.4f}s → {enhanced_results['avg_execution_time']:.4f}s ({comparison['execution_time_change']:+.4f}s)")
    
    print(f"\n🚀 Enhanced Features Available:")
    enhanced_features = [
        "✅ Multi-keyword scoring and ranking",
        "✅ Category-based analysis",
        "✅ Intersection detection",
        "✅ Region-specific boosts",
        "✅ Context extraction",
        "✅ Performance optimization",
        "✅ Configurable weights",
        "✅ Detailed match analysis"
    ]
    
    for feature in enhanced_features:
        print(f"   {feature}")
        comparison["enhanced_features"].append(feature)
    
    print(f"\n📈 Additional Insights:")
    print(f"   Categories detected: {len(enhanced_results['category_matches'])}")
    print(f"   Intersection detection: {enhanced_results['intersection_rate']*100:.1f}%")
    print(f"   Score distribution: {enhanced_results['score_distribution']}")
    
    # Quality verdict
    if comparison["accuracy_improvement"] > 0.1:
        verdict = "🟢 SIGNIFICANT IMPROVEMENT"
    elif comparison["accuracy_improvement"] > 0.05:
        verdict = "🟡 MODERATE IMPROVEMENT"
    elif comparison["accuracy_improvement"] > -0.05:
        verdict = "🟠 MAINTAINED PERFORMANCE"
    else:
        verdict = "🔴 PERFORMANCE DEGRADATION"
    
    print(f"\n🎖️  Overall Assessment: {verdict}")
    
    return comparison


def demonstrate_specific_use_cases():
    """Demonstrate specific multi-keyword use cases."""
    print("\n🎯 Specific Use Case Demonstrations")
    print("=" * 80)
    
    collector = EnhancedMultiKeywordCollector()
    
    # Use Case 1: UK AI + Insurance
    print("\n1️⃣  UK AI + Insurance Scenario")
    print("-" * 40)
    
    uk_article = {
        "title": "London Insurtech Startup Raises £50M for AI Platform",
        "content": "A London-based insurtech company has secured £50 million to develop artificial intelligence solutions for insurance fraud detection and automated claims processing in the Lloyd's market.",
        "region": "uk"
    }
    
    result = collector.analyze_multi_keywords(
        title=uk_article["title"],
        content=uk_article["content"],
        region=uk_article["region"]
    )
    
    print(f"   Score: {result.final_score:.3f}")
    print(f"   Categories: {list(result.category_scores.keys())}")
    print(f"   Region boost: {result.region_boost:.3f}")
    
    # Use Case 2: Multi-category intersection
    print("\n2️⃣  Multi-Category Intersection")
    print("-" * 40)
    
    intersection_article = {
        "title": "AI-Powered Fintech Platform for Healthcare Insurance",
        "content": "Machine learning algorithms process medical insurance claims through fintech payment systems with real-time fraud detection.",
        "region": "global"
    }
    
    result = collector.analyze_multi_keywords(
        title=intersection_article["title"],
        content=intersection_article["content"],
        region=intersection_article["region"]
    )
    
    print(f"   Score: {result.final_score:.3f}")
    print(f"   Categories: {list(result.category_scores.keys())}")
    print(f"   Intersection score: {result.intersection_score:.3f}")
    
    # Use Case 3: Advanced filtering
    print("\n3️⃣  Advanced Multi-Criteria Filtering")
    print("-" * 40)
    
    articles = [uk_article, intersection_article]
    filter_criteria = collector.create_ai_insurance_uk_filter()
    filtered = collector.filter_articles(articles, filter_criteria)
    
    print(f"   Articles filtered: {len(filtered)}")
    for article, analysis in filtered:
        print(f"     • {article['title'][:50]}... (Score: {analysis.final_score:.3f})")


def main():
    """Main comparison function."""
    print("🎯 Performance Comparison: Original vs Enhanced Multi-Keyword Collection")
    print("=" * 100)
    
    # Create test articles
    test_articles = create_test_articles()
    
    try:
        # Test original collector
        original_results = test_original_collector(test_articles)
        
        # Test enhanced collector
        enhanced_results = test_enhanced_collector(test_articles)
        
        # Compare results
        comparison = compare_collectors(original_results, enhanced_results)
        
        # Demonstrate specific use cases
        demonstrate_specific_use_cases()
        
        print(f"\n✅ Performance Comparison Complete!")
        print("=" * 100)
        
        # Final recommendations
        print(f"\n📋 Implementation Recommendations:")
        print(f"   1. 🔄 Replace SimpleCollector.is_ai_relevant() with enhanced scoring")
        print(f"   2. 🎯 Use category-based filtering for specific combinations")
        print(f"   3. 🌍 Implement region-specific boosts for local coverage")
        print(f"   4. 📊 Add intersection detection for high-value combinations")
        print(f"   5. ⚡ Enable performance mode for large-scale processing")
        print(f"   6. 📈 Implement real-time coverage monitoring")
        
        return {
            "original": original_results,
            "enhanced": enhanced_results,
            "comparison": comparison
        }
        
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
