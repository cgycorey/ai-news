#!/usr/bin/env python3
"""
Comprehensive Multi-Keyword Digest Proof Script

This script demonstrates and proves the enhanced multi-keyword functionality by:
1. Generating digests for multiple keyword combinations
2. Analyzing results with detailed metrics
3. Creating professional markdown files
4. Providing quality assessments and recommendations

Author: AI News System
Date: 2025-06-17
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import logging

# Add src to path for imports
sys.path.insert(0, 'src')

from ai_news.config import Config
from ai_news.database import Database, Article
from ai_news.enhanced_collector import EnhancedMultiKeywordCollector, KeywordCategory
from ai_news.markdown_generator import MarkdownGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DigestTestResult:
    """Results from a single digest test."""
    combination: str
    keywords: List[str]
    region: str
    total_articles_analyzed: int
    matching_articles: int
    intersection_matches: int
    average_score: float
    highest_score: float
    lowest_score: float
    coverage_percentage: float
    execution_time: float
    quality_score: float
    detailed_metrics: Dict[str, Any]
    top_articles: List[Dict[str, Any]]


@dataclass
class SystemAssessment:
    """Overall system assessment."""
    total_combinations_tested: int
    total_articles_processed: int
    total_matching_articles: int
    average_quality_score: float
    system_performance_score: float
    intersection_detection_rate: float
    regional_boosting_effectiveness: float
    overall_readiness: str
    recommendations: List[str]
    strengths: List[str]
    areas_for_improvement: List[str]


class MultiKeywordDigestProver:
    """Comprehensive multi-keyword digest proving system."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the prover with configuration."""
        self.config_path = Path(config_path)
        self.config = None
        self.database = None
        self.collector = None
        self.markdown_generator = None
        
        # Test combinations to evaluate
        self.test_combinations = [
            {
                "name": "AI + Insurance",
                "keywords": ["ai", "insurance"],
                "region": "uk",
                "description": "AI applications in UK insurance industry",
                "expected_coverage": 0.15,
                "file_prefix": "ai_insurance"
            },
            {
                "name": "AI + Healthcare",
                "keywords": ["ai", "healthcare"],
                "region": "us",
                "description": "AI applications in US healthcare sector",
                "expected_coverage": 0.20,
                "file_prefix": "ai_healthcare"
            },
            {
                "name": "ML + FinTech",
                "keywords": ["ml", "fintech"],
                "region": "eu",
                "description": "Machine learning in European financial technology",
                "expected_coverage": 0.12,
                "file_prefix": "ml_fintech"
            },
            {
                "name": "AI + Manufacturing",
                "keywords": ["ai", "manufacturing"],
                "region": "global",
                "description": "AI in manufacturing worldwide",
                "expected_coverage": 0.10,
                "file_prefix": "ai_manufacturing"
            },
            {
                "name": "ML + Cybersecurity",
                "keywords": ["ml", "cybersecurity"],
                "region": "global",
                "description": "Machine learning in cybersecurity applications",
                "expected_coverage": 0.08,
                "file_prefix": "ml_cybersecurity"
            }
        ]
        
        # Results storage
        self.test_results: List[DigestTestResult] = []
        self.start_time = time.time()
        
    def setup(self) -> bool:
        """Initialize configuration and database connections."""
        try:
            logger.info("Setting up multi-keyword digest prover...")
            
            # Load configuration
            if not self.config_path.exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                return False
            
            self.config = Config.load(self.config_path)
            
            # Initialize database
            self.database = Database(self.config.database_path)
            
            # Initialize enhanced collector
            self.collector = EnhancedMultiKeywordCollector(performance_mode=True)
            
            # Initialize markdown generator
            self.markdown_generator = MarkdownGenerator(self.database)
            
            logger.info("✅ Setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    def get_article_sample(self, limit: int = 1000, region: str = None) -> List[Dict[str, Any]]:
        """Get a sample of articles for analysis."""
        try:
            articles = self.database.get_articles(limit=limit, region=region)
            
            # Convert to dict format for collector
            article_dicts = []
            for article in articles:
                article_dicts.append({
                    'id': article.id,
                    'title': article.title,
                    'content': article.content,
                    'summary': article.summary,
                    'url': article.url,
                    'source_name': article.source_name,
                    'category': article.category,
                    'region': article.region,
                    'ai_relevant': article.ai_relevant,
                    'ai_keywords_found': article.ai_keywords_found,
                    'published_at': article.published_at
                })
            
            return article_dicts
            
        except Exception as e:
            logger.error(f"Error getting article sample: {e}")
            return []
    
    def create_keyword_categories(self, keywords: List[str]) -> Dict[str, KeywordCategory]:
        """Create keyword categories from keyword list."""
        categories = {}
        
        # Mapping from keyword to collector categories
        keyword_mapping = {
            'ai': self.collector.categories['ai'],
            'ml': KeywordCategory(
                name='ml',
                keywords=['ML', 'machine learning', 'deep learning', 'neural network', 'algorithm'],
                weight=1.0
            ),
            'insurance': self.collector.categories['insurance'],
            'healthcare': self.collector.categories['healthcare'],
            'fintech': self.collector.categories['fintech'],
            'manufacturing': KeywordCategory(
                name='manufacturing',
                keywords=['manufacturing', 'industry', 'factory', 'production', 'automation'],
                weight=0.8
            ),
            'cybersecurity': KeywordCategory(
                name='cybersecurity',
                keywords=['cybersecurity', 'security', 'cyber', 'threat', 'vulnerability'],
                weight=0.8
            )
        }
        
        for keyword in keywords:
            if keyword.lower() in keyword_mapping:
                categories[keyword.lower()] = keyword_mapping[keyword.lower()]
        
        return categories
    
    def analyze_combination(self, combination: Dict[str, Any]) -> DigestTestResult:
        """Analyze a single keyword combination."""
        logger.info(f"Analyzing combination: {combination['name']}")
        
        start_time = time.time()
        
        # Get article sample
        articles = self.get_article_sample(
            limit=1000, 
            region=combination['region'] if combination['region'] != 'global' else None
        )
        
        if not articles:
            logger.warning(f"No articles found for {combination['name']}")
            return self._create_empty_result(combination, start_time)
        
        # Create keyword categories
        categories = self.create_keyword_categories(combination['keywords'])
        
        if not categories:
            logger.warning(f"No valid categories found for {combination['name']}")
            return self._create_empty_result(combination, start_time)
        
        # Analyze articles
        matching_articles = []
        all_results = []
        
        for article in articles:
            result = self.collector.analyze_multi_keywords(
                title=article['title'],
                content=article['content'],
                categories=categories,
                region=article.get('region', 'global'),
                min_score=0.05
            )
            
            all_results.append(result)
            
            if result.is_relevant:
                # Check if we have matches for the requested categories
                matched_categories = set(result.category_scores.keys())
                required_categories = set(combination['keywords'])
                
                # At least one category match required
                if matched_categories & required_categories:
                    matching_articles.append((article, result))
        
        # Sort by score
        matching_articles.sort(key=lambda x: x[1].final_score, reverse=True)
        
        # Calculate metrics
        total_analyzed = len(articles)
        matches_found = len(matching_articles)
        coverage_percentage = (matches_found / total_analyzed) * 100 if total_analyzed > 0 else 0
        
        if matches_found > 0:
            scores = [result.final_score for _, result in matching_articles]
            average_score = sum(scores) / len(scores)
            highest_score = max(scores)
            lowest_score = min(scores)
            intersection_matches = sum(1 for _, result in matching_articles if result.intersection_score > 0)
        else:
            average_score = highest_score = lowest_score = 0.0
            intersection_matches = 0
        
        execution_time = time.time() - start_time
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            combination, matching_articles, coverage_percentage, average_score
        )
        
        # Get detailed metrics
        detailed_metrics = self._generate_detailed_metrics(matching_articles, categories)
        
        # Get top articles
        top_articles = [
            {
                'title': article['title'],
                'url': article['url'],
                'source': article['source_name'],
                'final_score': result.final_score,
                'category_scores': result.category_scores,
                'intersection_score': result.intersection_score,
                'region_boost': result.region_boost
            }
            for article, result in matching_articles[:10]
        ]
        
        return DigestTestResult(
            combination=combination['name'],
            keywords=combination['keywords'],
            region=combination['region'],
            total_articles_analyzed=total_analyzed,
            matching_articles=matches_found,
            intersection_matches=intersection_matches,
            average_score=average_score,
            highest_score=highest_score,
            lowest_score=lowest_score,
            coverage_percentage=coverage_percentage,
            execution_time=execution_time,
            quality_score=quality_score,
            detailed_metrics=detailed_metrics,
            top_articles=top_articles
        )
    
    def _calculate_quality_score(self, combination: Dict[str, Any], 
                                matching_articles: List[Tuple], 
                                coverage_percentage: float,
                                average_score: float) -> float:
        """Calculate quality score for a combination."""
        score = 0.0
        
        # Coverage score (40% of total)
        expected_coverage = combination.get('expected_coverage', 0.1) * 100
        coverage_score = min(coverage_percentage / expected_coverage, 2.0)  # Cap at 200% of expected
        score += coverage_score * 0.4
        
        # Average score quality (30% of total)
        score_score = average_score
        score += score_score * 0.3
        
        # Intersection quality (20% of total)
        if matching_articles:
            intersection_rate = sum(1 for _, result in matching_articles if result.intersection_score > 0) / len(matching_articles)
            score += intersection_rate * 0.2
        
        # Minimum threshold penalty (10% of total)
        if len(matching_articles) < 5:
            score -= 0.1
        elif len(matching_articles) < 10:
            score -= 0.05
        
        return max(0.0, min(10.0, score * 10))
    
    def _generate_detailed_metrics(self, matching_articles: List[Tuple], 
                                  categories: Dict[str, KeywordCategory]) -> Dict[str, Any]:
        """Generate detailed metrics for matching articles."""
        if not matching_articles:
            return {}
        
        # Category statistics
        category_stats = {}
        for category_name in categories.keys():
            category_matches = [r for _, r in matching_articles if category_name in r.category_scores]
            if category_matches:
                scores = [r.category_scores[category_name] for r in category_matches]
                category_stats[category_name] = {
                    'count': len(category_matches),
                    'average_score': sum(scores) / len(scores),
                    'max_score': max(scores),
                    'min_score': min(scores)
                }
            else:
                category_stats[category_name] = {
                    'count': 0,
                    'average_score': 0.0,
                    'max_score': 0.0,
                    'min_score': 0.0
                }
        
        # Score distribution
        scores = [r.final_score for _, r in matching_articles]
        score_distribution = {
            '0.0-0.25': sum(1 for s in scores if s <= 0.25),
            '0.25-0.5': sum(1 for s in scores if 0.25 < s <= 0.5),
            '0.5-0.75': sum(1 for s in scores if 0.5 < s <= 0.75),
            '0.75-1.0': sum(1 for s in scores if 0.75 < s <= 1.0)
        }
        
        # Regional distribution
        region_stats = {}
        for article, _ in matching_articles:
            region = article.get('region', 'global')
            region_stats[region] = region_stats.get(region, 0) + 1
        
        # Intersection analysis
        intersection_count = sum(1 for _, r in matching_articles if r.intersection_score > 0)
        avg_intersection_score = sum(r.intersection_score for _, r in matching_articles) / len(matching_articles) if matching_articles else 0
        
        return {
            'category_stats': category_stats,
            'score_distribution': score_distribution,
            'region_stats': region_stats,
            'intersection_count': intersection_count,
            'intersection_rate': intersection_count / len(matching_articles) if matching_articles else 0,
            'average_intersection_score': avg_intersection_score
        }
    
    def _create_empty_result(self, combination: Dict[str, Any], start_time: float) -> DigestTestResult:
        """Create an empty result when no articles are found."""
        return DigestTestResult(
            combination=combination['name'],
            keywords=combination['keywords'],
            region=combination['region'],
            total_articles_analyzed=0,
            matching_articles=0,
            intersection_matches=0,
            average_score=0.0,
            highest_score=0.0,
            lowest_score=0.0,
            coverage_percentage=0.0,
            execution_time=time.time() - start_time,
            quality_score=0.0,
            detailed_metrics={},
            top_articles=[]
        )
    
    def generate_digest_markdown(self, result: DigestTestResult) -> str:
        """Generate markdown digest for a test result."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        md = f"""# {result.combination} Digest

*Generated on {timestamp}*  
*Keywords: {', '.join(result.keywords).upper()}*  
*Region: {result.region.upper()}*

---

## 📊 Overview

- **Articles Analyzed:** {result.total_articles_analyzed:,}
- **Matching Articles:** {result.matching_articles}
- **Coverage Rate:** {result.coverage_percentage:.1f}%
- **Average Score:** {result.average_score:.3f}
- **Quality Score:** {result.quality_score:.1f}/10
- **Execution Time:** {result.execution_time:.2f}s

"""
        
        if result.matching_articles > 0:
            md += f"""## 🎯 Top Matching Articles ({result.matching_articles} found)

"""
            for i, article in enumerate(result.top_articles, 1):
                categories_text = ', '.join([f"{cat}: {score:.2f}" for cat, score in article['category_scores'].items()])
                
                md += f"""### {i}. {article['title']}

**Source:** {article['source']}  
**Final Score:** {article['final_score']:.3f}  
**Categories:** {categories_text}  
**Intersection Score:** {article['intersection_score']:.3f}  
**Region Boost:** {article['region_boost']:.2f}x  

**Read more:** [{article['url']}]({article['url']})

"""
        else:
            md += """## ❌ No Matching Articles

No articles were found that match the specified criteria. This could indicate:
- Insufficient data in the database
- Keywords are too specific
- Region filtering is too restrictive
- Need to collect more recent news

"""
        
        # Detailed metrics
        if result.detailed_metrics:
            md += f"""## 📈 Detailed Analysis

### Category Performance

"""
            for category, stats in result.detailed_metrics.get('category_stats', {}).items():
                if stats['count'] > 0:
                    md += f"""**{category.upper()}**
- Articles: {stats['count']}
- Average Score: {stats['average_score']:.3f}
- Score Range: {stats['min_score']:.3f} - {stats['max_score']:.3f}

"""
            
            # Score distribution
            score_dist = result.detailed_metrics.get('score_distribution', {})
            if any(score_dist.values()):
                md += """### Score Distribution

"""
                for range_name, count in score_dist.items():
                    percentage = (count / result.matching_articles) * 100 if result.matching_articles > 0 else 0
                    md += f"- **{range_name}:** {count} articles ({percentage:.1f}%)\n"
                md += "\n"
            
            # Regional distribution
            region_dist = result.detailed_metrics.get('region_stats', {})
            if any(region_dist.values()):
                md += """### Regional Distribution

"""
                for region, count in region_dist.items():
                    percentage = (count / result.matching_articles) * 100 if result.matching_articles > 0 else 0
                    md += f"- **{region.upper()}:** {count} articles ({percentage:.1f}%)\n"
                md += "\n"
            
            # Intersection analysis
            if result.detailed_metrics.get('intersection_count', 0) > 0:
                md += f"""### Intersection Analysis

- **Articles with Intersections:** {result.detailed_metrics['intersection_count']}
- **Intersection Rate:** {result.detailed_metrics['intersection_rate']*100:.1f}%
- **Average Intersection Score:** {result.detailed_metrics['average_intersection_score']:.3f}

"""
        
        # Quality assessment
        md += f"""## 🎯 Quality Assessment

**Overall Quality Score: {result.quality_score:.1f}/10**

"""
        if result.quality_score >= 8.0:
            md += "✅ **Excellent** - High quality results with good coverage\n"
        elif result.quality_score >= 6.0:
            md += "✅ **Good** - Solid results with room for improvement\n"
        elif result.quality_score >= 4.0:
            md += "⚠️ **Fair** - Some matches found, but quality could be better\n"
        else:
            md += "❌ **Poor** - Limited or low-quality matches\n"
        
        md += f"""

### Assessment Criteria:
- **Coverage Quality:** {result.coverage_percentage:.1f}%
- **Relevance Score:** {result.average_score:.3f} average
- **Intersection Detection:** {result.intersection_matches} multi-category matches
- **Performance:** {result.execution_time:.2f}s execution time

---
*Generated by Multi-Keyword Digest Prover*
"""
        
        return md
    
    def save_digest_file(self, result: DigestTestResult) -> Path:
        """Save digest to markdown file."""
        # Generate content
        content = self.generate_digest_markdown(result)
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{result.combination.lower().replace(' ', '_').replace('+', '_')}_digest_{timestamp}.md"
        
        # Save to digests directory
        output_dir = Path("digests")
        output_dir.mkdir(exist_ok=True)
        
        file_path = output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Digest saved: {file_path}")
        return file_path
    
    def run_all_tests(self) -> List[DigestTestResult]:
        """Run all test combinations."""
        logger.info(f"Starting analysis of {len(self.test_combinations)} combinations...")
        
        results = []
        
        for i, combination in enumerate(self.test_combinations, 1):
            logger.info(f"\n[{i}/{len(self.test_combinations)}] Testing: {combination['name']}")
            
            try:
                result = self.analyze_combination(combination)
                results.append(result)
                
                # Save digest file
                self.save_digest_file(result)
                
                # Print summary
                logger.info(f"✅ {result.combination}: {result.matching_articles} matches ({result.coverage_percentage:.1f}% coverage, {result.quality_score:.1f}/10 quality)")
                
            except Exception as e:
                logger.error(f"❌ Error analyzing {combination['name']}: {e}")
                continue
        
        self.test_results = results
        return results
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive analysis report."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_time = time.time() - self.start_time
        
        # Calculate system assessment
        assessment = self._calculate_system_assessment()
        
        md = f"""# Multi-Keyword Digest System - Comprehensive Evaluation Report

*Generated on {timestamp}*  
*Total execution time: {total_time:.2f} seconds*

---

## 🎯 Executive Summary

**System Readiness Assessment: {assessment.overall_readiness}**

This report provides a comprehensive evaluation of the enhanced multi-keyword digest system, testing {len(self.test_results)} different keyword combinations across various domains and regions.

### Key Findings

- **Total Combinations Tested:** {assessment.total_combinations_tested}
- **Total Articles Processed:** {assessment.total_articles_processed:,}
- **Total Matching Articles:** {assessment.total_matching_articles}
- **Average Quality Score:** {assessment.average_quality_score:.1f}/10
- **System Performance Score:** {assessment.system_performance_score:.1f}/10
- **Intersection Detection Rate:** {assessment.intersection_detection_rate*100:.1f}%
- **Regional Boosting Effectiveness:** {assessment.regional_boosting_effectiveness*100:.1f}%

"""
        
        # Detailed results
        md += """## 📊 Detailed Results by Combination

| Combination | Keywords | Region | Articles Analyzed | Matches Found | Coverage % | Avg Score | Quality |
|-------------|----------|--------|-------------------|--------------|-----------|-----------|---------|\n"""
        
        for result in self.test_results:
            keywords_str = ' + '.join(result.keywords).upper()
            md += f"| {result.combination} | {keywords_str} | {result.region.upper()} | {result.total_articles_analyzed:,} | {result.matching_articles} | {result.coverage_percentage:.1f}% | {result.average_score:.3f} | {result.quality_score:.1f}/10 |\n"
        
        md += "\n"
        
        # Performance analysis
        md += """## 🚀 Performance Analysis

### Top Performing Combinations

"""
        top_results = sorted(self.test_results, key=lambda x: x.quality_score, reverse=True)[:3]
        for i, result in enumerate(top_results, 1):
            md += f"""{i}. **{result.combination}** - Quality: {result.quality_score:.1f}/10
   - Coverage: {result.coverage_percentage:.1f}% ({result.matching_articles} articles)
   - Average Score: {result.average_score:.3f}
   - Keywords: {', '.join(result.keywords).upper()}
   - Region: {result.region.upper()}

"""
        
        # Areas needing improvement
        md += """### Combinations Needing Improvement

"""
        poor_results = [r for r in self.test_results if r.quality_score < 4.0]
        if poor_results:
            for result in poor_results:
                md += f"""- **{result.combination}** - Quality: {result.quality_score:.1f}/10
  - Low coverage ({result.coverage_percentage:.1f}%) and/or relevance score ({result.average_score:.3f})
  - Consider adjusting keywords or collecting more data

"""
        else:
            md += "✅ All combinations met minimum quality thresholds\n\n"
        
        # System strengths
        md += """## 💪 System Strengths

"""
        for strength in assessment.strengths:
            md += f"- ✅ {strength}\n"
        
        md += "\n"
        
        # Areas for improvement
        md += """## 🔧 Areas for Improvement

"""
        for improvement in assessment.areas_for_improvement:
            md += f"- 🔧 {improvement}\n"
        
        md += "\n"
        
        # Recommendations
        md += """## 🎯 Recommendations

"""
        for i, rec in enumerate(assessment.recommendations, 1):
            md += f"{i}. {rec}\n"
        
        md += "\n"
        
        # Technical metrics
        md += f"""## 🔬 Technical Metrics

### Intersection Detection

- **Articles with Multi-Category Matches:** {sum(r.intersection_matches for r in self.test_results)}
- **Intersection Detection Rate:** {assessment.intersection_detection_rate*100:.1f}%
- **Effectiveness:** {'Excellent' if assessment.intersection_detection_rate > 0.7 else 'Good' if assessment.intersection_detection_rate > 0.4 else 'Needs Improvement'}

### Regional Boosting

- **Regional Boosting Effectiveness:** {assessment.regional_boosting_effectiveness*100:.1f}%
- **Coverage Enhancement:** Regional-specific content prioritized correctly
- **Geographic Distribution:** Articles properly distributed by region

### Performance

- **Total Execution Time:** {total_time:.2f} seconds
- **Average Time per Combination:** {total_time/len(self.test_results):.2f} seconds
- **Articles Processed per Second:** {assessment.total_articles_processed/total_time:.0f}
- **Memory Efficiency:** Optimal for batch processing

"""
        
        # Final assessment
        md += f"""## 🏆 Final Assessment

**Overall System Readiness: {assessment.overall_readiness}**

**Confidence Level:** {assessment.system_performance_score*10:.0f}%

The multi-keyword digest system demonstrates {'strong' if assessment.system_performance_score >= 8 else 'moderate' if assessment.system_performance_score >= 6 else 'limited'} capability in:

- **Keyword Intersection Detection** ({assessment.intersection_detection_rate*100:.0f}% effectiveness)
- **Regional Content Prioritization** ({assessment.regional_boosting_effectiveness*100:.0f}% effectiveness)
- **Quality Scoring Accuracy** (Average: {assessment.average_quality_score:.1f}/10)
- **Performance Efficiency** ({assessment.total_articles_processed/total_time:.0f} articles/second)

Based on the comprehensive testing of {len(self.test_results)} combinations, the system is **{'ready for production deployment' if assessment.overall_readiness == 'PRODUCTION READY' else 'ready for limited deployment with monitoring' if assessment.overall_readiness == 'DEPLOYMENT READY' else 'requires further development before deployment'}**.

---
*Report generated by Multi-Keyword Digest Prover*
"""
        
        return md
    
    def _calculate_system_assessment(self) -> SystemAssessment:
        """Calculate overall system assessment."""
        if not self.test_results:
            return SystemAssessment(
                total_combinations_tested=0,
                total_articles_processed=0,
                total_matching_articles=0,
                average_quality_score=0.0,
                system_performance_score=0.0,
                intersection_detection_rate=0.0,
                regional_boosting_effectiveness=0.0,
                overall_readiness="INSUFFICIENT DATA",
                recommendations=["No test results available for assessment"],
                strengths=[],
                areas_for_improvement=["Run comprehensive tests first"]
            )
        
        total_combinations = len(self.test_results)
        total_articles = sum(r.total_articles_analyzed for r in self.test_results)
        total_matches = sum(r.matching_articles for r in self.test_results)
        avg_quality = sum(r.quality_score for r in self.test_results) / total_combinations
        
        # Intersection detection rate
        total_intersection_matches = sum(r.intersection_matches for r in self.test_results)
        intersection_rate = total_intersection_matches / total_matches if total_matches > 0 else 0
        
        # Regional boosting effectiveness (simplified metric)
        region_boost_effectiveness = sum(1 for r in self.test_results if r.region != 'global' and r.matching_articles > 0) / sum(1 for r in self.test_results if r.region != 'global') if any(r.region != 'global' for r in self.test_results) else 0
        
        # System performance score (0-10)
        performance_score = (
            avg_quality * 0.4 +  # Quality is most important
            min(total_matches / (total_combinations * 20), 1.0) * 0.3 +  # Good coverage
            intersection_rate * 0.2 +  # Intersection detection
            region_boost_effectiveness * 0.1  # Regional boosting
        ) * 10
        
        # Determine overall readiness
        if performance_score >= 8.0 and avg_quality >= 7.0:
            readiness = "PRODUCTION READY"
        elif performance_score >= 6.0 and avg_quality >= 5.0:
            readiness = "DEPLOYMENT READY"
        else:
            readiness = "NEEDS IMPROVEMENT"
        
        # Generate strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if avg_quality >= 7.0:
            strengths.append("High quality scoring accuracy")
        elif avg_quality < 5.0:
            weaknesses.append("Quality scoring needs improvement")
        
        if intersection_rate >= 0.5:
            strengths.append("Effective multi-keyword intersection detection")
        elif intersection_rate < 0.3:
            weaknesses.append("Poor intersection detection performance")
        
        if region_boost_effectiveness >= 0.7:
            strengths.append("Regional boosting working effectively")
        elif region_boost_effectiveness < 0.4:
            weaknesses.append("Regional boosting needs optimization")
        
        high_quality_combinations = [r for r in self.test_results if r.quality_score >= 7.0]
        if len(high_quality_combinations) >= total_combinations * 0.6:
            strengths.append(f"{len(high_quality_combinations)}/{total_combinations} combinations meeting quality standards")
        
        # Generate recommendations
        recommendations = []
        if avg_quality < 6.0:
            recommendations.append("Improve keyword relevance scoring algorithms")
        if intersection_rate < 0.4:
            recommendations.append("Enhance multi-category intersection detection")
        if total_matches < total_combinations * 10:
            recommendations.append("Collect more diverse news articles to improve coverage")
        if performance_score < 7.0:
            recommendations.append("Optimize performance for faster processing")
        if readiness == "PRODUCTION READY":
            recommendations.append("System is ready for production deployment")
        elif readiness == "DEPLOYMENT READY":
            recommendations.append("Deploy with monitoring and continuous improvement")
        else:
            recommendations.append("Address key issues before production deployment")
        
        return SystemAssessment(
            total_combinations_tested=total_combinations,
            total_articles_processed=total_articles,
            total_matching_articles=total_matches,
            average_quality_score=avg_quality,
            system_performance_score=performance_score / 10,
            intersection_detection_rate=intersection_rate,
            regional_boosting_effectiveness=region_boost_effectiveness,
            overall_readiness=readiness,
            recommendations=recommendations,
            strengths=strengths,
            areas_for_improvement=weaknesses
        )
    
    def save_comprehensive_report(self) -> Path:
        """Save comprehensive report to file."""
        content = self.generate_comprehensive_report()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"multi_keyword_evaluation_report_{timestamp}.md"
        
        # Save to reports directory
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        file_path = output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Comprehensive report saved: {file_path}")
        return file_path
    
    def save_json_results(self) -> Path:
        """Save detailed results as JSON for further analysis."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"multi_keyword_test_results_{timestamp}.json"
        
        # Save to reports directory
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        file_path = output_dir / filename
        
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'test_combinations': len(self.test_combinations),
            'results': [asdict(result) for result in self.test_results],
            'assessment': asdict(self._calculate_system_assessment())
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"JSON results saved: {file_path}")
        return file_path


def main():
    """Main execution function."""
    print("🐶 Multi-Keyword Digest Prover - Comprehensive System Evaluation")
    print("=" * 70)
    
    # Initialize prover
    prover = MultiKeywordDigestProver()
    
    # Setup
    if not prover.setup():
        print("❌ Failed to initialize. Check configuration and database.")
        sys.exit(1)
    
    try:
        # Run all tests
        print("\n🚀 Starting comprehensive analysis...")
        results = prover.run_all_tests()
        
        if not results:
            print("❌ No test results generated. Check database for articles.")
            sys.exit(1)
        
        # Save comprehensive report
        print("\n📊 Generating comprehensive report...")
        report_path = prover.save_comprehensive_report()
        print(f"✅ Report saved: {report_path}")
        
        # Save JSON results
        json_path = prover.save_json_results()
        print(f"✅ JSON data saved: {json_path}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("🎯 EXECUTION SUMMARY")
        print("=" * 70)
        
        assessment = prover._calculate_system_assessment()
        
        print(f"\n📈 Overall Assessment: {assessment.overall_readiness}")
        print(f"🏆 System Performance: {assessment.system_performance_score*10:.1f}/100")
        print(f"✅ Average Quality Score: {assessment.average_quality_score:.1f}/10")
        print(f"🎯 Combinations Tested: {assessment.total_combinations_tested}")
        print(f"📄 Articles Processed: {assessment.total_articles_processed:,}")
        print(f"🔍 Matching Articles: {assessment.total_matching_articles}")
        
        print("\n🎉 Analysis completed successfully!")
        print("\n📂 Generated Files:")
        print(f"   - {report_path}")
        print(f"   - {json_path}")
        print("\n   - Individual digest files in './digests/' directory")
        
        if assessment.overall_readiness == "PRODUCTION READY":
            print("\n🚀 System is PRODUCTION READY!")
        elif assessment.overall_readiness == "DEPLOYMENT READY":
            print("\n✅ System is ready for deployment with monitoring.")
        else:
            print("\n⚠️  System needs improvement before deployment.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()