#!/usr/bin/env python3
"""
Content Source Analysis Tool
Analyzes RSS feeds, regional content distribution, and identifies gaps
"""

import sqlite3
import json
import requests
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
from urllib.parse import urlparse
import time

@dataclass
class ContentGap:
    """Represents a content gap analysis."""
    category: str
    region: str
    missing_domains: List[str]
    recommended_feeds: List[Dict[str, Any]]
    severity: str

class ContentSourceAnalyzer:
    """Comprehensive content source analysis and gap identification."""

    def __init__(self, db_path: str = "ai_news.db", config_path: str = "config.json"):
        self.db_path = db_path
        self.config_path = config_path
        self.gaps = []
        self.analysis_results = {}

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete content source analysis."""
        print("📡 CONTENT SOURCE ANALYSIS")
        print("=" * 50)

        # 1. RSS Feed Audit
        self.audit_rss_feeds()

        # 2. Regional Content Distribution
        self.analyze_regional_distribution()

        # 3. Domain Coverage Analysis
        self.analyze_domain_coverage()

        # 4. Content Gap Identification
        self.identify_content_gaps()

        # 5. Generate recommendations
        self.generate_recommendations()

        return self.analysis_results

    def audit_rss_feeds(self) -> Dict[str, Any]:
        """Audit all RSS feeds by category and region."""
        print("\n🔍 RSS FEED AUDIT")
        print("-" * 30)

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"❌ Error reading config: {e}")
            return {}

        feed_analysis = {
            "total_feeds": 0,
            "enabled_feeds": 0,
            "by_category": defaultdict(int),
            "by_region": defaultdict(int),
            "feed_health": {},
            "categories": {},
            "problem_feeds": []
        }

        # Analyze feeds structure
        if 'regions' in config:
            # New format with regions
            for region_name, region_data in config['regions'].items():
                feeds = region_data.get('feeds', [])
                feed_analysis["by_region"][region_name] = len(feeds)
                
                for feed in feeds:
                    feed_analysis["total_feeds"] += 1
                    if feed.get('enabled', True):
                        feed_analysis["enabled_feeds"] += 1
                    
                    category = feed.get('category', 'unknown')
                    feed_analysis["by_category"][category] += 1
                    
                    # Store feed details
                    if category not in feed_analysis["categories"]:
                        feed_analysis["categories"][category] = []
                    feed_analysis["categories"][category].append({
                        "name": feed.get('name', 'Unknown'),
                        "url": feed.get('url', ''),
                        "enabled": feed.get('enabled', True),
                        "region": region_name
                    })
        else:
            # Legacy format
            feeds = config.get('feeds', [])
            feed_analysis["total_feeds"] = len(feeds)
            
            for feed in feeds:
                if feed.get('enabled', True):
                    feed_analysis["enabled_feeds"] += 1
                
                category = feed.get('category', 'unknown')
                feed_analysis["by_category"][category] += 1

        print(f"Total feeds: {feed_analysis['total_feeds']}")
        print(f"Enabled feeds: {feed_analysis['enabled_feeds']}")
        print(f"Enable rate: {feed_analysis['enabled_feeds']/feed_analysis['total_feeds']*100:.1f}%")

        print("\nFeeds by category:")
        for category, count in sorted(feed_analysis['by_category'].items()):
            print(f"  {category}: {count} feeds")

        print("\nFeeds by region:")
        for region, count in sorted(feed_analysis['by_region'].items()):
            print(f"  {region}: {count} feeds")

        self.analysis_results["rss_audit"] = feed_analysis
        return feed_analysis

    def analyze_regional_distribution(self) -> Dict[str, Any]:
        """Analyze actual content distribution by region in database."""
        print("\n🌍 REGIONAL CONTENT DISTRIBUTION")
        print("-" * 35)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        regional_analysis = {
            "total_articles": 0,
            "by_region": {},
            "ai_relevant_by_region": {},
            "source_distribution": {},
            "content_gaps": []
        }

        # Get overall stats
        cursor.execute("SELECT COUNT(*) FROM articles")
        regional_analysis["total_articles"] = cursor.fetchone()[0]

        # Regional distribution
        cursor.execute("""
            SELECT region, COUNT(*) as total,
                   SUM(CASE WHEN ai_relevant = 1 THEN 1 ELSE 0 END) as ai_relevant
            FROM articles
            GROUP BY region
            ORDER BY total DESC
        """)
        
        regions = cursor.fetchall()
        for region, total, ai_count in regions:
            regional_analysis["by_region"][region] = total
            regional_analysis["ai_relevant_by_region"][region] = ai_count
            ai_percentage = (ai_count / total * 100) if total > 0 else 0
            print(f"  {region}: {total} articles ({ai_count} AI relevant, {ai_percentage:.1f}%)")

        # Source distribution by region
        cursor.execute("""
            SELECT region, source_name, COUNT(*) as count
            FROM articles
            GROUP BY region, source_name
            ORDER BY region, count DESC
        """)
        
        source_data = cursor.fetchall()
        for region, source, count in source_data:
            if region not in regional_analysis["source_distribution"]:
                regional_analysis["source_distribution"][region] = {}
            regional_analysis["source_distribution"][region][source] = count

        # Identify regional content gaps
        expected_regions = {'us', 'uk', 'eu', 'global', 'asia', 'africa'}
        available_regions = set(regional_analysis["by_region"].keys())
        missing_regions = expected_regions - available_regions

        if missing_regions:
            regional_analysis["content_gaps"].extend([
                {"type": "missing_region", "region": region} 
                for region in missing_regions
            ])
            print(f"\n⚠️  Missing regions: {missing_regions}")

        # Check for low-content regions
        for region, count in regional_analysis["by_region"].items():
            if count < 50:  # Arbitrary threshold for "low content"
                regional_analysis["content_gaps"].append({
                    "type": "low_content",
                    "region": region,
                    "count": count
                })
                print(f"⚠️  Low content in {region}: {count} articles")

        conn.close()
        self.analysis_results["regional_distribution"] = regional_analysis
        return regional_analysis

    def analyze_domain_coverage(self) -> Dict[str, Any]:
        """Analyze which domains have good coverage vs gaps."""
        print("\n🏢 DOMAIN COVERAGE ANALYSIS")
        print("-" * 30)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        domain_analysis = {
            "total_domains": 0,
            "domain_distribution": {},
            "high_volume_domains": [],
            "low_volume_domains": [],
            "category_coverage": {},
            "missing_categories": []
        }

        # Domain distribution
        cursor.execute("""
            SELECT source_name, COUNT(*) as count,
                   SUM(CASE WHEN ai_relevant = 1 THEN 1 ELSE 0 END) as ai_count,
                   COUNT(DISTINCT region) as regions
            FROM articles
            GROUP BY source_name
            ORDER BY count DESC
        """)
        
        domains = cursor.fetchall()
        domain_analysis["total_domains"] = len(domains)

        print(f"Total domains: {domain_analysis['total_domains']}")
        print("\nTop 15 domains by volume:")
        for i, (domain, count, ai_count, regions) in enumerate(domains[:15], 1):
            ai_percentage = (ai_count / count * 100) if count > 0 else 0
            domain_analysis["domain_distribution"][domain] = {
                "total_articles": count,
                "ai_relevant": ai_count,
                "ai_percentage": ai_percentage,
                "regions": regions
            }
            print(f"  {i:2d}. {domain}: {count:4d} articles ({ai_count:3d} AI relevant, {ai_percentage:5.1f}%, {regions} regions)")
            
            if count >= 50:
                domain_analysis["high_volume_domains"].append(domain)
            elif count <= 5:
                domain_analysis["low_volume_domains"].append(domain)

        # Category coverage by domain
        cursor.execute("""
            SELECT source_name, category, COUNT(*) as count
            FROM articles
            WHERE category IS NOT NULL AND category != ''
            GROUP BY source_name, category
            ORDER BY source_name, count DESC
        """)
        
        category_data = cursor.fetchall()
        for domain, category, count in category_data:
            if domain not in domain_analysis["category_coverage"]:
                domain_analysis["category_coverage"][domain] = {}
            domain_analysis["category_coverage"][domain][category] = count

        print(f"\n📊 Domain Health Summary:")
        print(f"  High volume domains (≥50 articles): {len(domain_analysis['high_volume_domains'])}")
        print(f"  Low volume domains (≤5 articles): {len(domain_analysis['low_volume_domains'])}")
        print(f"  Average articles per domain: {domain_analysis['total_domains'] and len(domains) and sum(d[1] for d in domains) // len(domains) or 0}")

        conn.close()
        self.analysis_results["domain_coverage"] = domain_analysis
        return domain_analysis

    def identify_content_gaps(self) -> List[ContentGap]:
        """Identify specific content gaps, especially for healthcare/fintech."""
        print("\n🕳️  CONTENT GAP ANALYSIS")
        print("-" * 30)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Test critical categories that are failing
        critical_categories = {
            'healthcare': ['healthcare', 'medical', 'medicine', 'health tech', 'digital health', 'biotech'],
            'fintech': ['fintech', 'financial technology', 'banking', 'finance', 'payments', 'cryptocurrency', 'blockchain'],
            'manufacturing': ['manufacturing', 'automation', 'factory', 'industrial', 'supply chain'],
            'insurance': ['insurance', 'insurtech', 'risk management', 'underwriting']
        }

        gaps = []

        for category, keywords in critical_categories.items():
            print(f"\nAnalyzing {category} coverage:")
            
            category_stats = {
                'total_mentions': 0,
                'ai_relevant_mentions': 0,
                'by_region': defaultdict(int),
                'by_domain': defaultdict(int),
                'keyword_coverage': {}
            }

            for keyword in keywords:
                cursor.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN ai_relevant = 1 THEN 1 ELSE 0 END) as ai_relevant,
                           region, source_name
                    FROM articles
                    WHERE (title LIKE ? OR content LIKE ? OR summary LIKE ?)
                    GROUP BY region, source_name
                """, (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"))
                
                results = cursor.fetchall()
                keyword_total = sum(row[0] for row in results)
                keyword_ai = sum(row[1] for row in results)
                
                category_stats['total_mentions'] += keyword_total
                category_stats['ai_relevant_mentions'] += keyword_ai
                category_stats['keyword_coverage'][keyword] = keyword_total

                for total, ai_count, region, domain in results:
                    category_stats['by_region'][region] += total
                    category_stats['by_domain'][domain] += total

            coverage_percentage = (category_stats['ai_relevant_mentions'] / category_stats['total_mentions'] * 100) if category_stats['total_mentions'] > 0 else 0
            
            print(f"  Total mentions: {category_stats['total_mentions']}")
            print(f"  AI relevant: {category_stats['ai_relevant_mentions']} ({coverage_percentage:.1f}%)")
            print(f"  Keywords found: {', '.join([f'{k}: {v}' for k, v in category_stats['keyword_coverage'].items() if v > 0])}")
            
            # Check if this category has insufficient coverage
            if category_stats['total_mentions'] < 20:  # Arbitrary threshold
                severity = "CRITICAL" if category_stats['total_mentions'] < 5 else "HIGH"
                gaps.append(ContentGap(
                    category=category,
                    region="all",
                    missing_domains=[],
                    recommended_feeds=self.get_recommended_feeds_for_category(category),
                    severity=severity
                ))
                print(f"  ⚠️  INSUFFICIENT COVERAGE ({severity})")

            # Show top domains for this category
            if category_stats['by_domain']:
                print(f"  Top domains: {', '.join([f'{d}: {c}' for d, c in sorted(category_stats['by_domain'].items(), key=lambda x: x[1], reverse=True)[:3]])}")

        # Check for missing US region specifically (critical for user's use case)
        cursor.execute("SELECT COUNT(*) FROM articles WHERE region = 'us'")
        us_count = cursor.fetchone()[0]
        if us_count == 0:
            gaps.append(ContentGap(
                category="all",
                region="us",
                missing_domains=[],
                recommended_feeds=self.get_recommended_feeds_for_region("us"),
                severity="CRITICAL"
            ))
            print("\n🚨 CRITICAL: No US content found in database")

        conn.close()
        self.gaps = gaps
        self.analysis_results["content_gaps"] = gaps
        return gaps

    def get_recommended_feeds_for_category(self, category: str) -> List[Dict[str, Any]]:
        """Get recommended RSS feeds for specific categories."""
        recommendations = {
            'healthcare': [
                {
                    'name': 'STAT News',
                    'url': 'https://statnews.com/feed/',
                    'category': 'healthcare',
                    'region': 'us',
                    'description': 'Leading health & medicine news'
                },
                {
                    'name': 'Medscape Medical News',
                    'url': 'https://www.medscape.com/public/rss/news',
                    'category': 'healthcare',
                    'region': 'global',
                    'description': 'Medical news and insights'
                },
                {
                    'name': 'Fierce Healthcare',
                    'url': 'https://www.fiercehealthcare.com/rss.xml',
                    'category': 'healthcare',
                    'region': 'us',
                    'description': 'Healthcare industry news'
                },
                {
                    'name': 'Digital Health News',
                    'url': 'https://www.digitalhealth.net/rss/news',
                    'category': 'healthcare',
                    'region': 'uk',
                    'description': 'Digital health transformation'
                }
            ],
            'fintech': [
                {
                    'name': 'FinTech Futures',
                    'url': 'https://www.fintechfutures.com/rss.xml',
                    'category': 'fintech',
                    'region': 'global',
                    'description': 'FinTech industry news and analysis'
                },
                {
                    'name': 'American Banker',
                    'url': 'https://www.americanbanker.com/rss',
                    'category': 'fintech',
                    'region': 'us',
                    'description': 'Banking and financial services news'
                },
                {
                    'name': 'The Block',
                    'url': 'https://www.theblockcrypto.com/rss/',
                    'category': 'fintech',
                    'region': 'global',
                    'description': 'Crypto and blockchain news'
                },
                {
                    'name': 'FinTech Weekly',
                    'url': 'https://fintechweekly.com/rss.xml',
                    'category': 'fintech',
                    'region': 'eu',
                    'description': 'European FinTech news'
                }
            ]
        }
        return recommendations.get(category, [])

    def get_recommended_feeds_for_region(self, region: str) -> List[Dict[str, Any]]:
        """Get recommended RSS feeds for specific regions."""
        recommendations = {
            'us': [
                {
                    'name': 'TechCrunch',
                    'url': 'https://techcrunch.com/feed/',
                    'category': 'tech',
                    'region': 'us',
                    'description': 'US tech startup news'
                },
                {
                    'name': 'VentureBeat',
                    'url': 'https://venturebeat.com/feed/',
                    'category': 'tech',
                    'region': 'us',
                    'description': 'Transformative tech news'
                },
                {
                    'name': 'The Verge',
                    'url': 'https://www.theverge.com/rss/index.xml',
                    'category': 'tech',
                    'region': 'us',
                    'description': 'Technology and culture'
                }
            ]
        }
        return recommendations.get(region, [])

    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        print("\n💡 RECOMMENDATIONS")
        print("-" * 20)

        recommendations = []

        # RSS Feed Recommendations
        rss_audit = self.analysis_results.get("rss_audit", {})
        if rss_audit.get("enabled_feeds", 0) / rss_audit.get("total_feeds", 1) < 0.8:
            recommendations.append(
                f"Enable more RSS feeds - currently only {rss_audit.get('enabled_feeds', 0)}/{rss_audit.get('total_feeds', 0)} are active"
            )

        # Regional Content Recommendations
        regional_dist = self.analysis_results.get("regional_distribution", {})
        missing_regions = [gap["region"] for gap in regional_dist.get("content_gaps", []) if gap["type"] == "missing_region"]
        if missing_regions:
            recommendations.append(f"Add RSS feeds for missing regions: {', '.join(missing_regions)}")

        # Content Gap Recommendations
        for gap in self.gaps:
            if gap.severity == "CRITICAL":
                if gap.category != "all":
                    recommendations.append(
                        f"CRITICAL: Add {gap.category} RSS feeds - current coverage insufficient"
                    )
                else:
                    recommendations.append(
                        f"CRITICAL: Add {gap.region.upper()} regional RSS feeds - no content found"
                    )

        # Domain Coverage Recommendations
        domain_coverage = self.analysis_results.get("domain_coverage", {})
        low_volume_count = len(domain_coverage.get("low_volume_domains", []))
        if low_volume_count > 5:
            recommendations.append(
                f"Review {low_volume_count} low-volume domains - consider replacing with higher-quality sources"
            )

        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

        # Save detailed recommendations to file
        detailed_recommendations = {
            "summary": recommendations,
            "critical_gaps": [gap for gap in self.gaps if gap.severity == "CRITICAL"],
            "recommended_feeds": {
                gap.category: gap.recommended_feeds 
                for gap in self.gaps 
                if gap.recommended_feeds and gap.severity == "CRITICAL"
            }
        }

        with open("content_gap_recommendations.json", "w") as f:
            json.dump(detailed_recommendations, f, indent=2, default=str)

        print(f"\n📄 Detailed recommendations saved to content_gap_recommendations.json")
        return recommendations

def main():
    """Run content source analysis."""
    analyzer = ContentSourceAnalyzer()
    results = analyzer.run_full_analysis()
    
    print(f"\n✅ Content source analysis complete!")
    print(f"📊 Found {len(analyzer.gaps)} content gaps")

if __name__ == "__main__":
    main()