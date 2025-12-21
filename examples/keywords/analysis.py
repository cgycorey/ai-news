#!/usr/bin/env python3
"""
Keyword Analysis Tool
Analyzes keyword variations, synonyms, and configuration against sample content
"""

import sqlite3
import json
import re
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import itertools

@dataclass
class KeywordMatch:
    """Represents a keyword match in content."""
    keyword: str
    category: str
    article_id: int
    title: str
    context: str
    relevance_score: float
    position: str  # title, content, summary

@dataclass
class KeywordSuggestion:
    """Represents a keyword improvement suggestion."""
    original_keyword: str
    suggested_variations: List[str]
    reason: str
    evidence: Dict[str, Any]
    priority: str

class KeywordAnalyzer:
    """Comprehensive keyword analysis and optimization."""

    def __init__(self, db_path: str = "ai_news.db", config_path: str = "config.json"):
        self.db_path = db_path
        self.config_path = config_path
        self.keyword_matches = []
        self.suggestions = []
        self.analysis_results = {}

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete keyword analysis."""
        print("🔤 KEYWORD ANALYSIS")
        print("=" * 30)

        # 1. Load current keyword configuration
        self.load_keyword_config()

        # 2. Test keyword coverage and variations
        self.test_keyword_coverage()

        # 3. Analyze keyword combinations
        self.analyze_keyword_combinations()

        # 4. Test intersection detection
        self.test_intersection_detection()

        # 5. Suggest keyword improvements
        self.generate_keyword_suggestions()

        return self.analysis_results

    def load_keyword_config(self) -> Dict[str, Any]:
        """Load and analyze current keyword configuration."""
        print("\n📖 LOADING KEYWORD CONFIGURATION")
        print("-" * 35)

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"❌ Error reading config: {e}")
            return {}

        keyword_config = {
            "ai_keywords": set(),
            "industry_keywords": {},
            "total_keywords": 0,
            "keyword_categories": {}
        }

        # Extract keywords from config
        if 'regions' in config:
            for region_name, region_data in config['regions'].items():
                feeds = region_data.get('feeds', [])
                for feed in feeds:
                    ai_keywords = feed.get('ai_keywords', [])
                    industry_keywords = feed.get('industry_keywords', [])
                    
                    keyword_config["ai_keywords"].update(ai_keywords)
                    
                    category = feed.get('category', 'unknown')
                    if category not in keyword_config["industry_keywords"]:
                        keyword_config["industry_keywords"][category] = set()
                    keyword_config["industry_keywords"][category].update(industry_keywords)
        else:
            # Legacy format
            feeds = config.get('feeds', [])
            for feed in feeds:
                ai_keywords = feed.get('ai_keywords', [])
                industry_keywords = feed.get('industry_keywords', [])
                
                keyword_config["ai_keywords"].update(ai_keywords)
                
                category = feed.get('category', 'unknown')
                if category not in keyword_config["industry_keywords"]:
                    keyword_config["industry_keywords"][category] = set()
                keyword_config["industry_keywords"][category].update(industry_keywords)

        # Convert sets to lists for JSON compatibility
        keyword_config["ai_keywords"] = list(keyword_config["ai_keywords"])
        for category in keyword_config["industry_keywords"]:
            keyword_config["industry_keywords"][category] = list(keyword_config["industry_keywords"][category])

        # Count total keywords
        keyword_config["total_keywords"] = len(keyword_config["ai_keywords"])
        keyword_config["total_keywords"] += sum(len(kw_list) for kw_list in keyword_config["industry_keywords"].values())

        print(f"AI keywords: {len(keyword_config['ai_keywords'])}")
        for category, keywords in keyword_config["industry_keywords"].items():
            print(f"{category}: {len(keywords)} keywords")
        print(f"Total unique keywords: {keyword_config['total_keywords']}")

        self.analysis_results["keyword_config"] = keyword_config
        return keyword_config

    def test_keyword_coverage(self) -> Dict[str, Any]:
        """Test how well keywords match available content."""
        print("\n🎯 KEYWORD COVERAGE TEST")
        print("-" * 30)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Test keyword groups
        keyword_groups = {
            "AI": ["AI", "artificial intelligence", "machine learning", "ML", "deep learning", "neural network"],
            "Healthcare": ["healthcare", "medical", "medicine", "health", "hospital", "clinical", "pharma", "biotech"],
            "FinTech": ["fintech", "financial technology", "banking", "finance", "payments", "cryptocurrency", "blockchain", "trading"],
            "Manufacturing": ["manufacturing", "automation", "factory", "industrial", "supply chain", "production"],
            "Insurance": ["insurance", "insurtech", "risk management", "underwriting", "claims", "premium"]
        }

        coverage_results = {}
        total_articles = 0

        # Get total article count
        cursor.execute("SELECT COUNT(*) FROM articles")
        total_articles = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM articles WHERE ai_relevant = 1")
        total_ai_articles = cursor.fetchone()[0]

        print(f"Testing against {total_articles} total articles ({total_ai_articles} AI relevant)")

        for group_name, keywords in keyword_groups.items():
            group_results = {
                "keywords": {},
                "total_matches": 0,
                "unique_articles": set(),
                "coverage_percentage": 0
            }

            print(f"\n{group_name} keywords:")

            for keyword in keywords:
                cursor.execute("""
                    SELECT id, title, summary, content, 
                           CASE WHEN title LIKE ? THEN 'title'
                                WHEN summary LIKE ? THEN 'summary'
                                WHEN content LIKE ? THEN 'content'
                                ELSE 'none' END as position
                    FROM articles
                    WHERE (title LIKE ? OR summary LIKE ? OR content LIKE ?)
                    AND ai_relevant = 1
                    LIMIT 10
                """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"))

                results = cursor.fetchall()
                group_results["keywords"][keyword] = len(results)
                group_results["total_matches"] += len(results)

                for article_id, title, summary, content, position in results:
                    group_results["unique_articles"].add(article_id)
                    
                    # Store keyword match details
                    context = title if position == 'title' else (summary[:100] if position == 'summary' else content[:100])
                    self.keyword_matches.append(KeywordMatch(
                        keyword=keyword,
                        category=group_name,
                        article_id=article_id,
                        title=title,
                        context=context,
                        relevance_score=1.0,  # Basic relevance
                        position=position
                    ))

                unique_count = len(set(r[0] for r in results))
                print(f"  '{keyword}': {len(results)} matches ({unique_count} unique articles)")

            group_results["unique_articles"] = len(group_results["unique_articles"])
            group_results["coverage_percentage"] = (group_results["unique_articles"] / total_ai_articles * 100) if total_ai_articles > 0 else 0

            print(f"  Total unique articles: {group_results['unique_articles']} ({group_results['coverage_percentage']:.1f}% coverage)")

            coverage_results[group_name] = group_results

        conn.close()
        self.analysis_results["keyword_coverage"] = coverage_results
        return coverage_results

    def analyze_keyword_combinations(self) -> Dict[str, Any]:
        """Analyze specific keyword combinations that are failing."""
        print("\n🔗 KEYWORD COMBINATION ANALYSIS")
        print("-" * 35)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Test the failing combinations reported by user
        test_combinations = [
            {
                "name": "AI + Healthcare",
                "ai_keywords": ["AI", "artificial intelligence", "machine learning", "ML"],
                "industry_keywords": ["healthcare", "medical", "medicine", "health", "hospital"],
                "expected_regions": ["us", "global"]
            },
            {
                "name": "ML + FinTech", 
                "ai_keywords": ["ML", "machine learning", "deep learning", "neural network"],
                "industry_keywords": ["fintech", "financial technology", "banking", "finance", "payments"],
                "expected_regions": ["eu", "global"]
            },
            {
                "name": "AI + Manufacturing",
                "ai_keywords": ["AI", "artificial intelligence", "automation"],
                "industry_keywords": ["manufacturing", "factory", "industrial", "production"],
                "expected_regions": ["global"]
            },
            {
                "name": "AI + Insurance",
                "ai_keywords": ["AI", "artificial intelligence", "machine learning"],
                "industry_keywords": ["insurance", "insurtech", "risk management"],
                "expected_regions": ["uk", "global"]
            }
        ]

        combination_results = {}

        for combo in test_combinations:
            print(f"\nTesting: {combo['name']}")
            
            combo_result = {
                "name": combo["name"],
                "ai_keyword_matches": {},
                "industry_keyword_matches": {},
                "intersection_matches": 0,
                "intersection_articles": [],
                "regional_distribution": {},
                "success_rate": 0
            }

            # Test AI keywords
            for keyword in combo["ai_keywords"]:
                cursor.execute("""
                    SELECT COUNT(*) FROM articles
                    WHERE (title LIKE ? OR content LIKE ? OR summary LIKE ?)
                    AND ai_relevant = 1
                """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"))
                count = cursor.fetchone()[0]
                combo_result["ai_keyword_matches"][keyword] = count

            # Test industry keywords
            for keyword in combo["industry_keywords"]:
                cursor.execute("""
                    SELECT COUNT(*) FROM articles
                    WHERE (title LIKE ? OR content LIKE ? OR summary LIKE ?)
                    AND ai_relevant = 1
                """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"))
                count = cursor.fetchone()[0]
                combo_result["industry_keyword_matches"][keyword] = count

            # Test intersection (articles containing both types)
            ai_conditions = " OR ".join([f"(title LIKE '%{kw}%' OR content LIKE '%{kw}%' OR summary LIKE '%{kw}%')" for kw in combo["ai_keywords"]])
            industry_conditions = " OR ".join([f"(title LIKE '%{kw}%' OR content LIKE '%{kw}%' OR summary LIKE '%{kw}%')" for kw in combo["industry_keywords"]])

            intersection_query = f"""
                SELECT id, title, region, source_name, 
                       GROUP_CONCAT(DISTINCT CASE WHEN title LIKE '%{combo['ai_keywords'][0]}%' OR content LIKE '%{combo['ai_keywords'][0]}%' OR summary LIKE '%{combo['ai_keywords'][0]}%' THEN '{combo['ai_keywords'][0]}' WHEN title LIKE '%{combo['ai_keywords'][1] if len(combo['ai_keywords']) > 1 else combo['ai_keywords'][0]}%' OR content LIKE '%{combo['ai_keywords'][1] if len(combo['ai_keywords']) > 1 else combo['ai_keywords'][0]}%' OR summary LIKE '%{combo['ai_keywords'][1] if len(combo['ai_keywords']) > 1 else combo['ai_keywords'][0]}%' THEN '{combo['ai_keywords'][1] if len(combo['ai_keywords']) > 1 else combo['ai_keywords'][0]}' END) as ai_found,
                       GROUP_CONCAT(DISTINCT CASE WHEN title LIKE '%{combo['industry_keywords'][0]}%' OR content LIKE '%{combo['industry_keywords'][0]}%' OR summary LIKE '%{combo['industry_keywords'][0]}%' THEN '{combo['industry_keywords'][0]}' WHEN title LIKE '%{combo['industry_keywords'][1] if len(combo['industry_keywords']) > 1 else combo['industry_keywords'][0]}%' OR content LIKE '%{combo['industry_keywords'][1] if len(combo['industry_keywords']) > 1 else combo['industry_keywords'][0]}%' OR summary LIKE '%{combo['industry_keywords'][1] if len(combo['industry_keywords']) > 1 else combo['industry_keywords'][0]}%' THEN '{combo['industry_keywords'][1] if len(combo['industry_keywords']) > 1 else combo['industry_keywords'][0]}' END) as industry_found
                FROM articles
                WHERE ai_relevant = 1
                AND ({ai_conditions})
                AND ({industry_conditions})
                GROUP BY id, title, region, source_name
                LIMIT 10
            """

            try:
                cursor.execute(intersection_query)
                intersections = cursor.fetchall()
                combo_result["intersection_matches"] = len(intersections)
                combo_result["intersection_articles"] = intersections

                # Regional distribution of intersections
                for article in intersections:
                    region = article[2]
                    combo_result["regional_distribution"][region] = combo_result["regional_distribution"].get(region, 0) + 1

            except Exception as e:
                print(f"  Error in intersection query: {e}")
                combo_result["intersection_matches"] = 0

            # Calculate success rate
            ai_total = sum(combo_result["ai_keyword_matches"].values())
            industry_total = sum(combo_result["industry_keyword_matches"].values())
            
            if ai_total > 0 and industry_total > 0:
                # Success rate as percentage of intersection vs minimum of individual categories
                min_individual = min(ai_total, industry_total)
                combo_result["success_rate"] = (combo_result["intersection_matches"] / min_individual * 100) if min_individual > 0 else 0
            else:
                combo_result["success_rate"] = 0

            # Print results
            ai_total = sum(combo_result["ai_keyword_matches"].values())
            industry_total = sum(combo_result["industry_keyword_matches"].values())
            
            print(f"  AI keywords: {ai_total} total matches")
            for kw, count in combo_result["ai_keyword_matches"].items():
                if count > 0:
                    print(f"    '{kw}': {count}")
            
            print(f"  Industry keywords: {industry_total} total matches")
            for kw, count in combo_result["industry_keyword_matches"].items():
                if count > 0:
                    print(f"    '{kw}': {count}")
            
            print(f"  Intersection: {combo_result['intersection_matches']} articles")
            print(f"  Success rate: {combo_result['success_rate']:.1f}%")
            
            if combo_result["regional_distribution"]:
                print(f"  Regional distribution: {dict(combo_result['regional_distribution'])}")
            else:
                print(f"  Regional distribution: None (critical issue!)")

            # Identify issues
            if combo_result["intersection_matches"] == 0:
                if ai_total == 0 or industry_total == 0:
                    print(f"  ❌ CRITICAL: No base content for one or both keyword categories")
                else:
                    print(f"  ❌ CRITICAL: No intersection found despite individual keyword matches")
            elif combo_result["success_rate"] < 10:
                print(f"  ⚠️  POOR: Low intersection success rate")
            elif not combo_result["regional_distribution"]:
                print(f"  ⚠️  WARNING: No regional distribution in intersection")

            combination_results[combo["name"]] = combo_result

        conn.close()
        self.analysis_results["keyword_combinations"] = combination_results
        return combination_results

    def test_intersection_detection(self) -> Dict[str, Any]:
        """Test the effectiveness of current intersection detection."""
        print("\n🎯 INTERSECTION DETECTION TEST")
        print("-" * 35)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Test different intersection strategies
        strategies = [
            {
                "name": "Strict AND",
                "description": "Requires exact keyword matches in same article",
                "query_template": "title LIKE '%{ai_kw}%' AND title LIKE '%{ind_kw}%'"
            },
            {
                "name": "Loose OR",
                "description": "Any AI keyword AND any industry keyword in any field",
                "query_template": "(title LIKE '%{ai_kw}%' OR content LIKE '%{ai_kw}%') AND (title LIKE '%{ind_kw}%' OR content LIKE '%{ind_kw}%')"
            },
            {
                "name": "Semantic Approximation",
                "description": "Approximate matching with common variations",
                "query_template": "(title LIKE '%{ai_kw}%' OR title LIKE '%{ai_var}%') AND (title LIKE '%{ind_kw}%' OR title LIKE '%{ind_var}%')"
            }
        ]

        test_cases = [
            ("AI", "healthcare", "artificial intelligence", "medical"),
            ("ML", "fintech", "machine learning", "financial technology"),
            ("AI", "manufacturing", "automation", "factory")
        ]

        intersection_results = {}

        for ai_cat, ind_cat, ai_kw, ind_kw in test_cases:
            print(f"\nTesting {ai_cat} + {ind_cat}:")
            
            case_results = {}
            
            for strategy in strategies:
                try:
                    if strategy["name"] == "Semantic Approximation":
                        # Use variations for semantic test
                        ai_var = ai_kw.replace("artificial intelligence", "AI").replace("machine learning", "ML")
                        ind_var = ind_kw.replace("financial technology", "fintech").replace("medical", "healthcare")
                        query = strategy["query_template"].format(ai_kw=ai_kw, ai_var=ai_var, ind_kw=ind_kw, ind_var=ind_var)
                    else:
                        query = strategy["query_template"].format(ai_kw=ai_kw, ind_kw=ind_kw)
                    
                    full_query = f"SELECT COUNT(*) FROM articles WHERE ai_relevant = 1 AND {query}"
                    cursor.execute(full_query)
                    count = cursor.fetchone()[0]
                    
                    case_results[strategy["name"]] = count
                    print(f"  {strategy['name']}: {count} articles")
                    
                except Exception as e:
                    print(f"  {strategy['name']}: Error - {e}")
                    case_results[strategy["name"]] = 0
            
            # Analyze effectiveness
            strict_count = case_results.get("Strict AND", 0)
            loose_count = case_results.get("Loose OR", 0)
            
            if strict_count == 0 and loose_count > 0:
                print(f"  💡 Current intersection logic too strict - loose matching finds {loose_count} articles")
            elif strict_count > 0 and loose_count == strict_count:
                print(f"  ✅ Intersection logic working correctly")
            elif strict_count > 0 and loose_count > strict_count:
                improvement = ((loose_count - strict_count) / strict_count * 100) if strict_count > 0 else 0
                print(f"  💡 Loose matching improves results by {improvement:.1f}% ({strict_count} → {loose_count})")
            
            intersection_results[f"{ai_cat}+{ind_cat}"] = case_results

        conn.close()
        self.analysis_results["intersection_detection"] = intersection_results
        return intersection_results

    def generate_keyword_suggestions(self) -> List[KeywordSuggestion]:
        """Generate keyword improvement suggestions."""
        print("\n💡 KEYWORD IMPROVEMENT SUGGESTIONS")
        print("-" * 40)

        suggestions = []

        # Analyze keyword coverage results
        coverage = self.analysis_results.get("keyword_coverage", {})
        
        for category, results in coverage.items():
            coverage_pct = results.get("coverage_percentage", 0)
            keyword_matches = results.get("keywords", {})
            
            if coverage_pct < 10:  # Poor coverage
                # Find keywords with zero matches
                zero_keywords = [kw for kw, count in keyword_matches.items() if count == 0]
                
                if zero_keywords:
                    suggestions.append(KeywordSuggestion(
                        original_keyword=f"{category} keywords: {', '.join(zero_keywords)}",
                        suggested_variations=self.get_keyword_variations(category, zero_keywords),
                        reason=f"Zero matches found - keywords may be too specific or outdated",
                        evidence={"category": category, "coverage_percentage": coverage_pct, "zero_keywords": zero_keywords},
                        priority="CRITICAL" if coverage_pct == 0 else "HIGH"
                    ))

        # Analyze combination results
        combinations = self.analysis_results.get("keyword_combinations", {})
        
        for combo_name, combo_result in combinations.items():
            success_rate = combo_result.get("success_rate", 0)
            
            if success_rate < 10:  # Failing combinations
                ai_matches = combo_result.get("ai_keyword_matches", {})
                industry_matches = combo_result.get("industry_keyword_matches", {})
                
                # Find weak keywords
                weak_ai = [kw for kw, count in ai_matches.items() if count == 0]
                weak_industry = [kw for kw, count in industry_matches.items() if count == 0]
                
                if weak_ai or weak_industry:
                    suggestions.append(KeywordSuggestion(
                        original_keyword=f"{combo_name} combination",
                        suggested_variations=self.get_combination_improvements(combo_name, weak_ai, weak_industry),
                        reason=f"Combination failing with {success_rate:.1f}% success rate",
                        evidence={
                            "combination": combo_name,
                            "success_rate": success_rate,
                            "weak_ai_keywords": weak_ai,
                            "weak_industry_keywords": weak_industry
                        },
                        priority="CRITICAL" if success_rate == 0 else "HIGH"
                    ))

        # Sort suggestions by priority
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        suggestions.sort(key=lambda x: priority_order.get(x.priority, 4))

        # Print top suggestions
        print(f"\nTop 10 keyword improvements:")
        for i, suggestion in enumerate(suggestions[:10], 1):
            print(f"\n{i}. [{suggestion.priority}] {suggestion.original_keyword}")
            print(f"   Reason: {suggestion.reason}")
            if suggestion.suggested_variations:
                print(f"   Suggested: {', '.join(suggestion.suggested_variations[:3])}{'...' if len(suggestion.suggested_variations) > 3 else ''}")

        self.suggestions = suggestions
        self.analysis_results["keyword_suggestions"] = suggestions
        return suggestions

    def get_keyword_variations(self, category: str, zero_keywords: List[str]) -> List[str]:
        """Get suggested variations for keywords with zero matches."""
        variation_map = {
            "healthcare": {
                "medical": ["health tech", "digital health", "health IT", "medical technology"],
                "medicine": ["pharmaceutical", "drugs", "medication", "biopharma"],
                "hospital": ["healthcare system", "medical center", "clinic", "health system"],
                "pharma": ["pharmaceutical", "biotech", "life sciences", "drug discovery"],
                "biotech": ["biotechnology", "life sciences", "genomics", "biopharma"]
            },
            "fintech": {
                "fintech": ["financial technology", "digital banking", "online banking", "neobank"],
                "financial technology": ["fintech", "digital finance", "finance tech", "banking technology"],
                "banking": ["digital banking", "online banking", "mobile banking", "neobank"],
                "finance": ["financial services", "banking", "investment", "wealth management"],
                "payments": ["digital payments", "online payments", "mobile payments", "fintech"],
                "cryptocurrency": ["crypto", "bitcoin", "blockchain", "digital currency"],
                "blockchain": ["distributed ledger", "DLT", "crypto", "smart contracts"],
                "trading": ["investing", "stock market", "securities", "wealth management"]
            }
        }
        
        variations = []
        category_map = variation_map.get(category, {})
        
        for keyword in zero_keywords:
            if keyword.lower() in category_map:
                variations.extend(category_map[keyword.lower()])
            else:
                # Generic variations
                variations.extend([
                    keyword.replace(" ", ""),  # Remove spaces
                    keyword.replace("technology", "tech"),  # Common abbreviation
                    f"digital {keyword}",  # Digital transformation
                    f"online {keyword}"  # Online version
                ])
        
        return list(set(variations))  # Remove duplicates

    def get_combination_improvements(self, combo_name: str, weak_ai: List[str], weak_industry: List[str]) -> List[str]:
        """Get specific improvements for failing combinations."""
        if "Healthcare" in combo_name:
            return [
                "Add 'health tech', 'digital health' to industry keywords",
                "Add 'AI in healthcare', 'healthcare AI' as compound keywords",
                "Include medical device, pharma, biotechnology variations",
                "Add telemedicine, remote care, digital therapeutics"
            ]
        elif "FinTech" in combo_name:
            return [
                "Add 'digital banking', 'neobank', 'challenger bank' to industry keywords",
                "Include 'regtech', 'wealthtech', 'insurtech' variations",
                "Add cryptocurrency, DeFi, blockchain technology terms",
                "Include 'AI in finance', 'algorithmic trading' variations"
            ]
        else:
            return [
                "Expand keyword variations for both AI and industry categories",
                "Add compound keywords that combine AI + industry terms",
                "Include semantic variations and synonyms",
                "Consider industry-specific terminology and jargon"
            ]

    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        print("\n📋 KEYWORD ANALYSIS REPORT")
        print("=" * 40)

        report_lines = []
        
        # Summary statistics
        keyword_config = self.analysis_results.get("keyword_config", {})
        coverage = self.analysis_results.get("keyword_coverage", {})
        combinations = self.analysis_results.get("keyword_combinations", {})
        
        total_keywords = keyword_config.get("total_keywords", 0)
        failing_combinations = len([c for c in combinations.values() if c.get("success_rate", 0) < 10])
        
        report_lines.append(f"Total Keywords Configured: {total_keywords}")
        report_lines.append(f"Keyword Categories Analyzed: {len(coverage)}")
        report_lines.append(f"Failing Combinations: {failing_combinations}")
        report_lines.append(f"Keyword Suggestions Generated: {len(self.suggestions)}")
        
        # Critical issues
        critical_suggestions = [s for s in self.suggestions if s.priority == "CRITICAL"]
        if critical_suggestions:
            report_lines.append(f"\n🚨 CRITICAL ISSUES ({len(critical_suggestions)}):")
            for i, suggestion in enumerate(critical_suggestions[:5], 1):
                report_lines.append(f"  {i}. {suggestion.reason}")
        
        # Key findings
        report_lines.append(f"\n🔍 KEY FINDINGS:")
        
        # Best performing categories
        if coverage:
            best_category = max(coverage.items(), key=lambda x: x[1].get("coverage_percentage", 0))
            report_lines.append(f"  • Best coverage: {best_category[0]} ({best_category[1].get('coverage_percentage', 0):.1f}%)")
        
        # Worst performing categories
        if coverage:
            worst_category = min(coverage.items(), key=lambda x: x[1].get("coverage_percentage", 0))
            report_lines.append(f"  • Worst coverage: {worst_category[0]} ({worst_category[1].get('coverage_percentage', 0):.1f}%)")
        
        # Intersection analysis
        if combinations:
            avg_success = sum(c.get("success_rate", 0) for c in combinations.values()) / len(combinations)
            report_lines.append(f"  • Average combination success rate: {avg_success:.1f}%")
        
        return "\n".join(report_lines)

def main():
    """Run keyword analysis."""
    analyzer = KeywordAnalyzer()
    results = analyzer.run_full_analysis()
    report = analyzer.generate_report()
    
    # Save results to JSON
    with open("keyword_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Keyword analysis complete!")
    print(f"📊 Results saved to keyword_analysis_results.json")
    print(f"💡 Generated {len(analyzer.suggestions)} keyword suggestions")

if __name__ == "__main__":
    main()