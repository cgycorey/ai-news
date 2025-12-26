"""Entity-aware topic digest generator.

This module provides enhanced topic digest generation with:
- Entity-aware matching (better than keywords)
- Entity insights (shows related companies/products)
- Smart grouping (by companies/products)
- Relevance scoring

Zero flags needed - works transparently.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta

from .enhanced_topic_digest import EnhancedTopicDigest, ScoredArticle
from .database import Database

logger = logging.getLogger(__name__)


class EntityAwareDigestGenerator:
    """Generate entity-aware topic digests.
    
    This generator enhances topic digests with:
    - Entity-based matching (instead of just keywords)
    - Entity insights section
    - Grouping by companies/products
    - Relevance scores per article
    
    Zero flags needed - automatic enhancement.
    """
    
    def __init__(self, database: Database):
        """Initialize the digest generator.
        
        Args:
            database: Database instance
        """
        self.database = database
        self.enhanced_digest = EnhancedTopicDigest(database)
        self._test_article_ids = self._load_test_article_ids()
        logger.info(f"EntityAwareDigestGenerator initialized (filtered {len(self._test_article_ids)} test articles)")
    
    def _load_test_article_ids(self) -> set:
        """Load IDs of test articles to filter from digests."""
        import sqlite3
        test_ids = set()
        try:
            conn = sqlite3.connect(str(self.database.db_path))
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM articles 
                WHERE url LIKE '%.test%'
                   OR url LIKE '%test.com%'
                   OR url LIKE '%test-%'
                   OR url LIKE '%-test.%'
                   OR url LIKE '%success-test%'
                   OR url LIKE '%unique-%-test%'
                   OR url LIKE '%example.com%'
                   OR url LIKE '%localhost%'
                   OR title = 'Test'
            ''')
            test_ids = {row[0] for row in cursor.fetchall()}
            conn.close()
        except Exception as e:
            logger.warning(f"Could not load test article IDs: {e}")
        return test_ids
    
    def generate_topic_digest(self, 
                              topics: List[str], 
                              days: int,
                              ai_only: bool = True,
                              group_by_entity: str = 'company') -> str:
        """Generate entity-aware topic digest with spaCy hybrid.
        
        Combines entity tag matching + spaCy semantic analysis for maximum coverage.
        - Entity tags: Fast, precise matching
        - spaCy: Semantic understanding (GPT → LLM connection)
        
        Args:
            topics: List of topic strings
            days: Number of days to look back
            ai_only: Only include AI-relevant articles
            group_by_entity: Entity type to group by ('company', 'product', etc.)
            
        Returns:
            Complete markdown digest
        """
        logger.info(f"Generating hybrid entity+spaCy digest for topics: {topics}")
        
        # Step 1: Get entity-tagged articles
        entity_scored = self.enhanced_digest.find_articles_for_topic(
            topics=topics,
            days=days,
            ai_only=ai_only,
            min_confidence=0.05
        )
        logger.info(f"Entity matching found {len(entity_scored)} articles")
        
        # Step 2: Get spaCy-scored articles (for semantic coverage)
        spacy_scored = self._get_spacy_articles(topics, days, ai_only)
        logger.info(f"spaCy matching found {len(spacy_scored)} articles")
        
        # Step 3: Combine and deduplicate
        combined = self._combine_scored_articles(entity_scored, spacy_scored)
        logger.info(f"Combined total: {len(combined)} unique articles")
        
        # Step 4: Filter out test articles
        if self._test_article_ids:
            filtered = [sa for sa in combined if sa.article.id not in self._test_article_ids]
            filtered_count = len(combined) - len(filtered)
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} test articles")
            combined = filtered
        
        if not combined:
            return self._generate_empty_digest(topics, days)
        
        if not combined:
            return self._generate_empty_digest(topics, days)
        
        # Generate entity insights
        entity_insights = self.enhanced_digest.generate_entity_insights(combined)
        
        # Group articles by entity
        grouped_articles = self.enhanced_digest.group_by_entity(
            combined, 
            entity_type=group_by_entity
        )
        
        # Generate digest
        digest = self._generate_digest_content(
            topics=topics,
            days=days,
            scored_articles=combined,
            entity_insights=entity_insights,
            grouped_articles=grouped_articles,
            group_by_entity=group_by_entity
        )
        
        return digest
    
    def _generate_empty_digest(self, topics: List[str], days: int) -> str:
        """Generate empty digest when no articles found."""
        topics_str = ", ".join(topics)
        return f"""# Topic Analysis: {topics_str} (Last {days} Days)

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*  
**Topics:** {topics_str}  
**Method:** Entity-aware matching

---

*No articles found for '{topics_str}' in the last {days} days.*

**Tips:**
- Try different topic keywords
- Increase the days parameter
- Check if articles have been tagged with entities
"""
    
    def _generate_digest_content(self,
                                 topics: List[str],
                                 days: int,
                                 scored_articles: List[ScoredArticle],
                                 entity_insights: Dict[str, Any],
                                 grouped_articles: Dict[str, List[ScoredArticle]],
                                 group_by_entity: str) -> str:
        """Generate the digest content."""
        topics_str = ", ".join(topics)
        
        # Header
        digest = f"""# Topic Analysis: {topics_str} (Last {days} Days)

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*  
**Topics:** {topics_str}  
**Method:** Entity-aware matching with enhanced insights

"""
        
        # Entity Insights Section
        digest += self._generate_entity_insights_section(entity_insights)
        
        digest += "\n---\n\n"
        
        # Grouped Articles Section
        digest += self._generate_grouped_articles_section(
            grouped_articles,
            group_by_entity,
            scored_articles
        )
        
        # All Articles Section (chronological)
        digest += self._generate_all_articles_section(scored_articles)
        
        # Statistics
        digest += self._generate_statistics_section(scored_articles)
        
        return digest
    
    def _generate_entity_insights_section(self, entity_insights: Dict[str, Any]) -> str:
        """Generate entity insights section."""
        section = "## 📊 Entity Insights\n\n"
        
        # Companies
        if entity_insights.get('companies'):
            section += "**Companies mentioned:**\n"
            for company, count in list(entity_insights['companies'].items())[:10]:
                section += f"- {company} ({count} article{'' if count == 1 else 's'})\n"
            section += "\n"
        
        # Products
        if entity_insights.get('products'):
            section += "**Products mentioned:**\n"
            for product, count in list(entity_insights['products'].items())[:10]:
                section += f"- {product} ({count} article{'' if count == 1 else 's'})\n"
            section += "\n"
        
        # Technologies
        if entity_insights.get('technologies'):
            section += "**Technologies mentioned:**\n"
            for tech, count in list(entity_insights['technologies'].items())[:10]:
                section += f"- {tech} ({count} article{'' if count == 1 else 's'})\n"
            section += "\n"
        
        # People
        if entity_insights.get('people'):
            section += "**People mentioned:**\n"
            for person, count in list(entity_insights['people'].items())[:10]:
                section += f"- {person} ({count} article{'' if count == 1 else 's'})\n"
            section += "\n"
        
        return section
    
    def _generate_grouped_articles_section(self,
                                          grouped_articles: Dict[str, List[ScoredArticle]],
                                          group_by_entity: str,
                                          all_articles: List[ScoredArticle]) -> str:
        """Generate grouped articles section."""
        section = f"## 🏢 Grouped by {group_by_entity.title()}\n\n"
        
        # Sort groups by article count (descending)
        sorted_groups = sorted(
            grouped_articles.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for entity_name, articles in sorted_groups:
            section += f"### {entity_name} ({len(articles)} article{'' if len(articles) == 1 else 's'})\n\n"
            
            for i, scored_article in enumerate(articles[:5], 1):  # Max 5 per group
                section += self._format_article(scored_article, i)
            
            if len(articles) > 5:
                section += f"*... and {len(articles) - 5} more article{'' if len(articles) - 5 == 1 else 's'}*\n"
            
            section += "\n"
        
        return section
    
    def _generate_all_articles_section(self, scored_articles: List[ScoredArticle]) -> str:
        """Generate all articles section (chronological)."""
        section = "## 📰 All Articles (Chronological)\n\n"
        
        # Sort by published date (newest first)
        def sort_by_date(sa):
            if sa.article.published_at:
                # Handle timezone-aware datetimes
                if sa.article.published_at.tzinfo is not None:
                    return sa.article.published_at.replace(tzinfo=None)
                return sa.article.published_at
            return datetime.min
        
        sorted_articles = sorted(scored_articles, key=sort_by_date, reverse=True)
        
        for i, scored_article in enumerate(sorted_articles, 1):
            section += self._format_article(scored_article, i)
        
        return section
    
    def _format_article(self, scored_article: ScoredArticle, index: int) -> str:
        """Format a single article."""
        article = scored_article.article
        confidence_pct = scored_article.confidence * 100
        
        # Get entity tags (only if article has ID)
        entity_tags = []
        if article.id is not None:
            entity_tags = self.database.get_article_entity_tags(article.id)
        
        entities_str = ", ".join([f"{tag['entity_text']} ({tag['entity_type']})" 
                                       for tag in entity_tags[:5]])  # Max 5
        if len(entity_tags) > 5:
            entities_str += f" ... and {len(entity_tags) - 5} more"
        
        # Format published date
        published_str = article.published_at.strftime('%Y-%m-%d %H:%M') if article.published_at else "Unknown"
        
        # Build article entry
        md = f"""#### {index}. {article.title}
 **Source:** {article.source_name} | **Published:** {published_str}  
 **Relevance:** {confidence_pct:.0f}% | **Entities:** {entities_str if entities_str else 'None'}
 
 {article.summary or article.content[:300] if article.content else ''}
 
 **Read more:** [{article.url}]({article.url})
 
 """
        return md
    
    def _generate_statistics_section(self, scored_articles: List[ScoredArticle]) -> str:
        """Generate statistics section."""
        total = len(scored_articles)
        high_confidence = sum(1 for sa in scored_articles if sa.confidence >= 0.7)
        avg_confidence = sum(sa.confidence for sa in scored_articles) / total if total > 0 else 0
        
        section = "---\n\n"
        section += "## 📈 Statistics\n\n"
        section += f"- **Total articles:** {total}\n"
        section += f"- **High confidence (≥70%):** {high_confidence} ({high_confidence/total*100:.1f}%)\n" if total > 0 else ""
        section += f"- **Average confidence:** {avg_confidence:.1%}\n" if total > 0 else ""
        
        return section
    
    def _get_spacy_articles(self, topics: List[str], days: int, ai_only: bool) -> List[ScoredArticle]:
        """Get articles scored by spaCy semantic analysis."""
        try:
            from .spacy_digest_analyzer import create_spacy_digest_analyzer
            
            analyzer = create_spacy_digest_analyzer(
                cache_db_path=str(self.database.db_path),
                ttl_hours=6
            )
            
            if not analyzer or not analyzer._spacy_available:
                logger.warning("spaCy not available")
                return []
            
            # Get articles from database
            all_articles = self.database.get_articles(limit=5000, ai_only=ai_only)
            
            # Filter by date
            cutoff = datetime.now() - timedelta(days=days)
            recent_articles = []
            for a in all_articles:
                if not a.published_at:
                    recent_articles.append(a)
                else:
                    if a.published_at.tzinfo:
                        article_date = a.published_at.astimezone(None).replace(tzinfo=None)
                    else:
                        article_date = a.published_at
                    if article_date >= cutoff:
                        recent_articles.append(a)
            
            # Convert to dict format for spaCy
            articles_dict = [
                {
                    'id': a.id,
                    'title': a.title,
                    'content': a.content or '',
                    'summary': a.summary or '',
                    'url': a.url,
                    'source_name': a.source_name,
                    'author': a.author,
                    'published_at': a.published_at,
                    'category': a.category,
                    'ai_relevant': a.ai_relevant,
                    'ai_keywords_found': a.ai_keywords_found or []
                }
                for a in recent_articles
            ]
            
            # Analyze with spaCy
            spacy_results = analyzer.analyze(
                articles=articles_dict,
                topics=topics,
                days=days,
                use_and_logic=False
            )
            
            # Convert spaCy ScoredArticle to enhanced_topic_digest ScoredArticle
            scored = []
            for spacy_sa in spacy_results:
                # spacy_sa.article is a dict, get the ID
                article_id = spacy_sa.article.get('id')
                if not article_id:
                    continue
                
                # Find original Article object
                article = next((a for a in recent_articles if a.id == article_id), None)
                if article:
                    scored.append(ScoredArticle(
                        article=article,
                        confidence=spacy_sa.confidence,
                        matched_entities=[]  # spaCy doesn't provide entity match details
                    ))
            
            return scored
            
        except Exception as e:
            logger.error(f"Error getting spaCy articles: {e}")
            return []
    
    def _combine_scored_articles(self, entity_articles: List[ScoredArticle], 
                                 spacy_articles: List[ScoredArticle]) -> List[ScoredArticle]:
        """Combine entity-tagged and spaCy-scored articles, removing duplicates.
        
        Uses entity score when available, falls back to spaCy score.
        Prefer entity-tagged articles (higher precision).
        """
        seen_ids = set()
        combined = []
        
        # Add entity-tagged articles first (higher precision)
        for sa in entity_articles:
            if sa.article.id not in seen_ids:
                seen_ids.add(sa.article.id)
                combined.append(sa)
        
        # Add spaCy articles not already seen
        for sa in spacy_articles:
            if sa.article.id not in seen_ids:
                seen_ids.add(sa.article.id)
                # Use spaCy confidence but mark as spaCy-sourced
                combined.append(sa)
        
        # Sort by confidence
        combined.sort(key=lambda x: x.confidence, reverse=True)
        return combined


def create_entity_aware_digest_generator(database: Database) -> EntityAwareDigestGenerator:
    """Factory function to create EntityAwareDigestGenerator.
    
    Args:
        database: Database instance
        
    Returns:
        EntityAwareDigestGenerator instance
    """
    return EntityAwareDigestGenerator(database)
