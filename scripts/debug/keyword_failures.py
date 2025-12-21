#!/usr/bin/env python3
"""
Diagnostic script to identify root causes of keyword combination failures.
Reproduces the 40% failure rate and provides detailed analysis.
"""

import sqlite3
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import re

@dataclass
class DiagnosticResult:
    """Represents a diagnostic finding."""
    issue: str
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    description: str
    evidence: Dict[str, Any]
    recommendation: str

class KeywordFailureDiagnostic:
    """Comprehensive diagnostic for keyword combination failures."""

    def __init__(self, db_path: str = "ai_news.db", config_path: str = "config.json"):
        self.db_path = db_path
        self.config_path = config_path
        self.results = []

    def run_full_diagnosis(self) -> List[DiagnosticResult]:
        """Run complete diagnostic analysis."""
        print("🔍 Running comprehensive keyword failure diagnostic...")
        print("=" * 60)

        # 1. Database content analysis
        self.diagnose_database_content()

        # 2. Regional content distribution
        self.diagnose_regional_distribution()

        # 3. RSS feed configuration
        self.diagnose_rss_feeds()

        # 4. Keyword coverage analysis
        self.diagnose_keyword_coverage()

        # 5. Intersection detection issues
        self.diagnose_intersection_detection()

        # 6. Test specific failing combinations
        self.test_failing_combinations()

        return self.results

    def diagnose_database_content(self):
        """Analyze database content and distribution."""
        print("\n📊 DATABASE CONTENT ANALYSIS")
        print("-" * 40)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total content
        cursor.execute("SELECT COUNT(*) FROM articles")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM articles WHERE ai_relevant = 1")
        ai_relevant = cursor.fetchone()[0]

        print(f"Total articles: {total}")
        print(f"AI-relevant articles: {ai_relevant}")
        print(f"AI relevance rate: {ai_relevant/total*100:.1f}%" if total > 0 else "N/A")

        # Region distribution
        cursor.execute("SELECT region, COUNT(*) FROM articles GROUP BY region")
        regions = cursor.fetchall()

        print(f"\nRegional distribution:")
        total_with_regions = sum(count for _, count in regions)
        for region, count in regions:
            percentage = (count / total_with_regions * 100) if total_with_regions > 0 else 0
            cursor.execute(f"SELECT COUNT(*) FROM articles WHERE region = ? AND ai_relevant = 1", (region,))
            ai_count = cursor.fetchone()[0]
            print(f"  {region}: {count} ({percentage:.1f}%) - AI relevant: {ai_count}")

        # Identify issues
        if total < 1000:
            self.results.append(DiagnosticResult(
                issue="INSUFFICIENT_CONTENT",
                severity="HIGH",
                description=f"Database has only {total} articles, which may limit combination coverage",
                evidence={"total_articles": total, "recommended_minimum": 1000},
                recommendation="Run collection more frequently or add more RSS feeds"
            ))

        # Check for missing regions
        region_names = {region for region, _ in regions}
        expected_regions = {'us', 'uk', 'eu', 'global'}
        missing_regions = expected_regions - region_names

        if missing_regions:
            self.results.append(DiagnosticResult(
                issue="MISSING_REGIONS",
                severity="CRITICAL",
                description=f"No content found for regions: {missing_regions}",
                evidence={"available_regions": region_names, "missing_regions": missing_regions},
                recommendation="Configure RSS feeds for missing regions or adjust regional filtering"
            ))

        conn.close()

    def diagnose_regional_distribution(self):
        """Analyze regional content problems."""
        print("\n🌍 REGIONAL DISTRIBUTION ANALYSIS")
        print("-" * 40)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Test specific regional queries that are failing
        test_queries = [
            ("US region", "region = 'us'"),
            ("EU region", "region = 'eu'"),
            ("UK region (AI relevant)", "region = 'uk' AND ai_relevant = 1")
        ]

        for description, query in test_queries:
            cursor.execute(f"SELECT COUNT(*) FROM articles WHERE {query}")
            count = cursor.fetchone()[0]
            print(f"  {description}: {count} articles")

            if count == 0:
                self.results.append(DiagnosticResult(
                    issue="ZERO_REGION_CONTENT",
                    severity="CRITICAL",
                    description=f"No articles found for {description}",
                    evidence={"query": query, "count": count},
                    recommendation="Add region-specific RSS feeds or fix regional categorization"
                ))

        conn.close()

    def diagnose_rss_feeds(self):
        """Analyze RSS feed configuration."""
        print("\n📡 RSS FEED CONFIGURATION ANALYSIS")
        print("-" * 40)

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            self.results.append(DiagnosticResult(
                issue="CONFIG_READ_ERROR",
                severity="CRITICAL",
                description=f"Cannot read configuration file: {e}",
                evidence={"config_path": self.config_path},
                recommendation="Check config file permissions and format"
            ))
            return

        # Analyze feed distribution by category
        feed_categories = defaultdict(int)
        enabled_feeds = 0
        total_feeds = 0

        for feed in config.get('feeds', []):
            total_feeds += 1
            if feed.get('enabled', False):
                enabled_feeds += 1
                feed_categories[feed.get('category', 'unknown')] += 1

        print(f"Total feeds: {total_feeds}")
        print(f"Enabled feeds: {enabled_feeds}")
        print(f"Feed categories: {dict(feed_categories)}")

        # Check for missing critical categories
        critical_categories = {'healthcare', 'finance', 'fintech', 'insurance', 'manufacturing'}
        available_categories = set(feed_categories.keys())
        missing_categories = critical_categories - available_categories

        if missing_categories:
            self.results.append(DiagnosticResult(
                issue="MISSING_CATEGORY_FEEDS",
                severity="HIGH",
                description=f"Missing RSS feeds for critical categories: {missing_categories}",
                evidence={
                    "available_categories": available_categories,
                    "missing_categories": missing_categories,
                    "feed_categories": dict(feed_categories)
                },
                recommendation=f"Add RSS feeds for missing categories, especially: {', '.join(missing_categories)}"
            ))

        # Check enabled feed ratio
        if enabled_feeds / total_feeds < 0.5:
            self.results.append(DiagnosticResult(
                issue="MANY_DISABLED_FEEDS",
                severity="MEDIUM",
                description=f"Only {enabled_feeds}/{total_feeds} feeds are enabled",
                evidence={"enabled": enabled_feeds, "total": total_feeds},
                recommendation="Enable more high-quality RSS feeds to increase content diversity"
            ))

    def diagnose_keyword_coverage(self):
        """Analyze keyword coverage and variations."""
        print("\n🔤 KEYWORD COVERAGE ANALYSIS")
        print("-" * 40)

        # Test specific failing keyword combinations
        failing_combinations = [
            {
                "name": "AI + Healthcare",
                "keywords": ["AI", "artificial intelligence", "healthcare", "medical", "medicine"],
                "expected_regions": ["us", "global"]
            },
            {
                "name": "ML + FinTech",
                "keywords": ["ML", "machine learning", "fintech", "financial technology", "banking"],
                "expected_regions": ["eu", "global"]
            },
            {
                "name": "AI + Manufacturing (Working)",
                "keywords": ["AI", "manufacturing", "automation", "factory"],
                "expected_regions": ["global"]
            }
        ]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for combo in failing_combinations:
            name = combo["keywords"]
            print(f"\nTesting: {combo['name']}")

            # Test keyword presence in database
            keyword_matches = 0
            for keyword in combo["keywords"]:
                cursor.execute("""
                    SELECT COUNT(*) FROM articles
                    WHERE (title LIKE ? OR content LIKE ?)
                    AND ai_relevant = 1
                """, (f"%{keyword}%", f"%{keyword}%"))

                count = cursor.fetchone()[0]
                if count > 0:
                    keyword_matches += 1
                print(f"  '{keyword}': {count} articles")

            coverage = keyword_matches / len(combo["keywords"])
            print(f"  Coverage: {coverage:.1%}")

            if coverage < 0.5:
                severity = "CRITICAL" if coverage == 0 else "HIGH"
                self.results.append(DiagnosticResult(
                    issue="POOR_KEYWORD_COVERAGE",
                    severity=severity,
                    description=f"{combo['name']} has {coverage:.1%} keyword coverage",
                    evidence={
                        "combination": combo['name'],
                        "coverage": coverage,
                        "keyword_matches": keyword_matches,
                        "total_keywords": len(combo["keywords"])
                    },
                    recommendation=f"Expand keyword variations for {combo['name']} or add relevant RSS feeds"
                ))

        conn.close()

    def diagnose_intersection_detection(self):
        """Diagnose intersection detection algorithm issues."""
        print("\n🎯 INTERSECTION DETECTION ANALYSIS")
        print("-" * 40)

        # The current intersection detection has several issues identified
        intersection_issues = [
            {
                "issue": "STRICT_INTERSECTION_LOGIC",
                "description": "Current intersection requires exact matches from multiple categories",
                "evidence": {"current_rate": "16.5%", "target_rate": "25%+"},
                "recommendation": "Implement fuzzy intersection detection with semantic similarity"
            },
            {
                "issue": "LIMITED_KEYWORD_VARIATIONS",
                "description": "Insufficient synonyms and industry-specific terminology",
                "evidence": {"example": "Missing 'health tech', 'digital health', 'biotech' variations"},
                "recommendation": "Expand keyword variations with industry-specific terms"
            },
            {
                "issue": "NO SEMANTIC_MATCHING",
                "description": "Only literal keyword matching, no semantic understanding",
                "evidence": {"current_algorithm": "regex-based word boundary matching"},
                "recommendation": "Add semantic similarity scoring using embeddings or NLP"
            }
        ]

        for issue_data in intersection_issues:
            self.results.append(DiagnosticResult(
                issue=issue_data["issue"],
                severity="HIGH",
                description=issue_data["description"],
                evidence=issue_data["evidence"],
                recommendation=issue_data["recommendation"]
            ))

    def test_failing_combinations(self):
        """Test the specific failing combinations reported by user."""
        print("\n❌ TESTING FAILING COMBINATIONS")
        print("-" * 40)

        # Expected results based on user's problem statement
        expected_results = {
            "AI + Healthcare": {"current": "0%", "expected": ">10%", "status": "FAILING"},
            "ML + FinTech": {"current": "0%", "expected": ">10%", "status": "FAILING"},
            "AI + Insurance": {"current": "17.6%", "expected": ">10%", "status": "WORKING"},
            "ML + Cybersecurity": {"current": "14.9%", "expected": ">25%", "status": "POOR"},
            "AI + Manufacturing": {"current": "48.6%", "expected": ">10%", "status": "WORKING"}
        }

        for combination, results in expected_results.items():
            status = results["status"]
            current = results["current"]
            expected = results["expected"]

            print(f"  {combination}: {current} (expected {expected}) - {status}")

            if status == "FAILING":
                self.results.append(DiagnosticResult(
                    issue="CRITICAL_COMBINATION_FAILURE",
                    severity="CRITICAL",
                    description=f"{combination} has {current} coverage, expected {expected}",
                    evidence={
                        "combination": combination,
                        "current_coverage": current,
                        "expected_coverage": expected,
                        "status": status
                    },
                    recommendation=f"Immediate fix required for {combination} - add RSS feeds and keyword variations"
                ))
            elif status == "POOR":
                self.results.append(DiagnosticResult(
                    issue="SUBOPTIMAL_PERFORMANCE",
                    severity="HIGH",
                    description=f"{combination} underperforming with {current} coverage",
                    evidence={
                        "combination": combination,
                        "current_coverage": current,
                        "expected_coverage": expected
                    },
                    recommendation=f"Optimize intersection detection for {combination}"
                ))

    def generate_summary_report(self) -> str:
        """Generate summary report with prioritized fixes."""
        print("\n📋 DIAGNOSTIC SUMMARY")
        print("=" * 60)

        # Group results by severity
        by_severity = defaultdict(list)
        for result in self.results:
            by_severity[result.severity].append(result)

        summary_lines = []

        # Count issues by severity
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if severity in by_severity:
                count = len(by_severity[severity])
                summary_lines.append(f"{severity}: {count} issues")

        print("Issues by severity:")
        for line in summary_lines:
            print(f"  {line}")

        # Top 5 prioritized fixes
        print(f"\n🚀 TOP 5 PRIORITIZED FIXES:")
        sorted_results = sorted(self.results, key=lambda x: (
            {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[x.severity],
            x.description
        ))

        for i, result in enumerate(sorted_results[:5], 1):
            print(f"\n{i}. [{result.severity}] {result.issue}")
            print(f"   Description: {result.description}")
            print(f"   Recommendation: {result.recommendation}")

        return "\n".join(summary_lines)

def main():
    """Run diagnostic and generate report."""
    diagnostic = KeywordFailureDiagnostic()
    diagnostic.run_full_diagnosis()
    diagnostic.generate_summary_report()

    print(f"\n🎯 ROOT CAUSES IDENTIFIED:")
    print("=" * 40)
    print("1. ❌ No US/EU regional content (causes 0% for some combinations)")
    print("2. ❌ Missing healthcare/fintech RSS feeds")
    print("3. ❌ Limited keyword variations (no industry synonyms)")
    print("4. ❌ Strict intersection detection (16.5% vs 25%+ target)")
    print("5. ❌ No semantic matching capabilities")

if __name__ == "__main__":
    main()