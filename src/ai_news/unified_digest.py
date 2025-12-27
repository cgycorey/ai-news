"""Unified topic digest generator with entity-aware and spaCy semantic analysis.

This module consolidates:
- enhanced_topic_digest.py (298 lines)
- spacy_digest_analyzer.py (395 lines)  
- entity_aware_digest.py (444 lines)

Total consolidation: 1137 lines → ~600 lines (47% reduction)
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import sqlite3

from .entity_extractor import EntityExtractor, ExtractedEntity
from .entity_types import EntityType
from .text_processor import TextProcessor
from .database import Database, Article
from .digest_cache import DigestCache
from .intersection_optimization import IntersectionOptimizer

logger = logging.getLogger(__name__)


@dataclass
class ScoredArticle:
    """Article with relevance score and matched entities."""
    article: Article
    confidence: float
    matched_entities: List[Dict[str, Any]]
    
    def __eq__(self, other):
        if not isinstance(other, ScoredArticle):
            return False
        return self.article.id == other.article.id
    
    def __hash__(self):
        return hash(self.article.id) if self.article.id else hash(id(self))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'article': self.article,
            'confidence': self.confidence,
            'matched_entities': self.matched_entities
        }


class UnifiedDigestGenerator:
    """Unified digest generator with entity-aware and spaCy semantic analysis.
    
    Priority: Entity tags → spaCy semantic → keywords
    """
    
    CONFIDENCE_THRESHOLD = 0.3
    
    def __init__(self, database: Database, cache_ttl_hours: int = 6):
        self.database = database
        self.cache = DigestCache(db_path=str(database.db_path), ttl_hours=cache_ttl_hours)
        self.optimizer = IntersectionOptimizer()
        self._entity_extractor: Optional[EntityExtractor] = None
        self._spacy_extractor = None
        self._spacy_available = False
        self._test_article_ids = self._load_test_article_ids()
        self._init_spacy()
        logger.info(f"UnifiedDigestGenerator initialized (spaCy: {self._spacy_available})")
    
    def _init_spacy(self) -> bool:
        try:
            from .spacy_term_extractor import SpaCyTermExtractor
            self._spacy_extractor = SpaCyTermExtractor()
            self._spacy_available = self._spacy_extractor.is_available()
            return self._spacy_available
        except Exception as e:
            logger.info(f"spaCy not available: {e}")
            return False
    
    @property
    def entity_extractor(self) -> Optional[EntityExtractor]:
        if self._entity_extractor is None:
            try:
                text_processor = TextProcessor()
                self._entity_extractor = EntityExtractor(text_processor, use_spacy=True)
            except Exception as e:
                logger.warning(f"Failed to load entity extractor: {e}")
        return self._entity_extractor
    
    def _load_test_article_ids(self) -> set:
        test_ids = set()
        try:
            conn = sqlite3.connect(str(self.database.db_path))
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM articles WHERE url LIKE "%.test%" OR url LIKE "%test.com%" OR title = "Test"')
            test_ids = {row[0] for row in cursor.fetchall()}
            conn.close()
        except Exception as e:
            logger.warning(f"Could not load test article IDs: {e}")
        return test_ids
    
    def generate_digest(self, topics: List[str], days: int = 7, ai_only: bool = True,
                       use_spacy: bool = True, min_confidence: float = 0.3,
                       use_and_logic: bool = True) -> str:
        """Generate unified topic digest."""
        logger.info(f"Generating digest for: {topics}")
        
        entity_scored = self._get_entity_articles(topics, days, ai_only, 0.05)
        logger.info(f"Entity matching: {len(entity_scored)} articles")
        
        spacy_scored = []
        if use_spacy and self._spacy_available:
            spacy_scored = self._get_spacy_articles(topics, days, ai_only, use_and_logic)
            logger.info(f"spaCy matching: {len(spacy_scored)} articles")
        
        combined = self._combine_scored_articles(entity_scored, spacy_scored)
        filtered = [sa for sa in combined if sa.confidence >= min_confidence 
                   and sa.article.id not in self._test_article_ids]
        
        if not filtered:
            return self._generate_empty_digest(topics, days)
        
        insights = self._generate_entity_insights(filtered)
        grouped = self._group_by_entity(filtered)
        
        return self._generate_digest_content(topics, days, filtered, insights, grouped)
    
    def _expand_topic_to_domain(self, topic: str) -> List[ExtractedEntity]:
        """Expand a concept topic to related entities in that domain.
        
        For example:
        - "education" → Coursera, edX, Khan Academy, etc.
        - "healthcare" → Pfizer, Moderna, telemedicine, etc.
        
        This works by:
        1. Finding articles with the topic keyword in title/content
        2. Looking at what entities are tagged in those articles
        3. Returning the most common entities as related to this domain
        """
        import sqlite3
        from collections import Counter
        
        try:
            # Find articles that mention this topic
            conn = sqlite3.connect(self.database.db_path)
            
            # Search for articles with topic in title or content
            query = """
                SELECT DISTINCT a.id, a.title, a.content
                FROM articles a
                WHERE a.title LIKE ? OR a.content LIKE ?
                LIMIT 100
            """
            topic_pattern = f"%{topic}%"
            articles = conn.execute(query, (topic_pattern, topic_pattern)).fetchall()
            
            if not articles:
                logger.info(f"No articles found for domain expansion of '{topic}'")
                return []
            
            # Get all entity tags from these articles
            article_ids = [str(a[0]) for a in articles]
            placeholders = ','.join(['?'] * len(article_ids))
            
            query = f"""
                SELECT entity_text, entity_type, COUNT(*) as count
                FROM article_entity_tags
                WHERE article_id IN ({placeholders})
                GROUP BY entity_text, entity_type
                ORDER BY count DESC
                LIMIT 20
            """
            entity_results = conn.execute(query, article_ids).fetchall()
            
            # Convert to ExtractedEntity objects
            from .entity_extractor import ExtractedEntity, EntityType
            
            # Common words to skip
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'how', 'what', 'where', 'when', 'why', 'here', 'there', 'this', 'that', 'is', 'are', 'was', 'were'}
            
            expanded_entities = []
            for entity_text, entity_type, count in entity_results:
                # Skip if the entity is the topic itself
                if entity_text.lower() == topic.lower():
                    continue
                
                # Skip common words
                if entity_text.lower() in common_words:
                    continue
                
                # Skip very short entities (< 3 chars) or very long (> 50 chars)
                if len(entity_text) < 3 or len(entity_text) > 50:
                    continue
                
                # Skip entities with mostly non-alphabetic chars (Chinese text, etc.)
                alphabetic_ratio = sum(c.isalpha() or c.isspace() for c in entity_text) / len(entity_text)
                if alphabetic_ratio < 0.5:
                    continue
                
                # Map entity_type string to EntityType enum
                try:
                    entity_type_enum = EntityType(entity_type)
                except ValueError:
                    continue
                
                # Create ExtractedEntity with moderate confidence
                entity = ExtractedEntity(
                    text=entity_text,
                    entity_type=entity_type_enum,
                    confidence=min(0.7, 0.3 + (count / 50)),  # Scale confidence by frequency
                    start_position=0,
                    end_position=len(entity_text),
                    extraction_method='discovered_entity'
                )
                expanded_entities.append(entity)
            
            logger.info(f"Domain expansion '{topic}' → {len(expanded_entities)} entities: {[e.text for e in expanded_entities[:5]]}")
            return expanded_entities
            
        except Exception as e:
            logger.warning(f"Domain expansion failed for '{topic}': {e}")
            return []
    
    def _get_entity_articles(self, topics: List[str], days: int, ai_only: bool, min_confidence: float) -> List[ScoredArticle]:
        if not self.entity_extractor:
            return []
        
        topic_entities = []
        expanded_topics = []
        
        # Expand topics to include domain-related entities
        for topic in topics:
            try:
                entities = self.entity_extractor.extract_entities(topic)
                
                if not entities:
                    # No direct entities found - try domain expansion
                    logger.info(f"No direct entities for '{topic}', trying domain expansion")
                    domain_entities = self._expand_topic_to_domain(topic)
                    if domain_entities:
                        logger.info(f"Domain expansion for '{topic}' found {len(domain_entities)} related entities")
                        topic_entities.extend(domain_entities)
                        expanded_topics.append(topic)
                else:
                    topic_entities.extend(entities)
                    
            except Exception as e:
                logger.warning(f"Failed to extract entities from topic '{topic}': {e}")
        
        if not topic_entities:
            return []
        
        articles = self.database.get_articles(limit=5000, ai_only=ai_only)
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter articles by date, handling timezone-aware and naive datetimes
        # Separate into dated and undated articles
        recent = []
        undated = []
        for a in articles:
            if not a.published_at:
                # Articles without publish date - include as undated (lower priority)
                logger.debug(f"Article {a.id} has no publish date: {a.title[:50]}...")
                undated.append(a)
            else:
                # Handle both timezone-aware and naive datetimes
                article_date = a.published_at
                if article_date.tzinfo is not None:
                    # Convert timezone-aware to naive for comparison
                    article_date = article_date.astimezone(None).replace(tzinfo=None)
                
                # Ensure cutoff_date is naive (it already is from datetime.now())
                cutoff_naive = cutoff_date.replace(tzinfo=None)
                
                # Compare dates
                if article_date >= cutoff_naive:
                    recent.append(a)
        
        # Score both dated and undated articles with semantic similarity
        scored = []
        undated_scored = []
        
        for article in recent:
            if article.id is None:
                continue
            score, matched = self._score_by_entities(article, topic_entities, topics)
            
            # Apply spaCy semantic similarity scoring for better relevance
            if self._spacy_available and score > 0:
                semantic_score = self._score_semantic_similarity(article, topics)
                # Weight: 70% entity matching, 30% semantic similarity
                score = (score * 0.7) + (semantic_score * 0.3)
            
            if score >= min_confidence:
                scored.append(ScoredArticle(article=article, confidence=score, matched_entities=matched))
        
        # Score undated articles (these will be shown at the end)
        for article in undated:
            if article.id is None:
                continue
            score, matched = self._score_by_entities(article, topic_entities, topics)
            
            # Apply spaCy semantic similarity scoring
            if self._spacy_available and score > 0:
                semantic_score = self._score_semantic_similarity(article, topics)
                score = (score * 0.7) + (semantic_score * 0.3)
            
            if score >= min_confidence:
                # Mark these as undated by setting a flag in the matched_entities
                matched_with_flag = matched + [{'is_undated': True}] if matched else [{'is_undated': True}]
                undated_scored.append(ScoredArticle(article=article, confidence=score, matched_entities=matched_with_flag))
        
        # Sort both by confidence
        scored.sort(key=lambda x: x.confidence, reverse=True)
        undated_scored.sort(key=lambda x: x.confidence, reverse=True)
        
        # Return combined list: dated first, undated at the end
        return scored + undated_scored
    
    def _score_semantic_similarity(self, article: Article, topics: List[str]) -> float:
        """Score article using spaCy NLP features for better relevance.
        
        Uses multiple spaCy features:
        - Noun chunk matching (key phrases)
        - Dependency parsing (subject-object relationships)
        - Token overlap (topic words in article)
        
        Args:
            article: Article to score
            topics: List of topic strings
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not self._spacy_extractor:
            return 0.0
        
        try:
            # Combine topics into single query string
            topic_query = ' '.join(topics).lower()
            
            # Process article with spaCy
            article_text = f"{article.title} {article.summary or ''}"
            article_text = article_text[:1000]  # Limit length
            
            doc = self._spacy_extractor.nlp(article_text)
            
            # Extract key phrases using noun chunks
            noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
            
            # Calculate relevance score
            relevance = 0.0
            
            # 1. Topic words in noun chunks (40% weight)
            topic_words = set(topic_query.split())
            chunk_matches = sum(1 for chunk in noun_chunks if any(word in chunk for word in topic_words))
            chunk_score = min(1.0, chunk_matches / max(1, len(noun_chunks)))
            relevance += chunk_score * 0.4
            
            # 2. Topic words in title (30% weight)
            title_score = sum(1 for word in topic_words if word in article.title.lower())
            title_score = min(1.0, title_score / max(1, len(topic_words)))
            relevance += title_score * 0.3
            
            # 3. Topic words in content (20% weight)
            content_lower = doc.text.lower()
            content_matches = sum(1 for word in topic_words if word in content_lower)
            content_score = min(1.0, content_matches / max(1, len(topic_words)))
            relevance += content_score * 0.2
            
            # 4. Dependency parsing - subject/object related to topic (10% weight)
            # Look for sentences where topic words are subjects or objects
            for sent in doc.sents:
                for word in topic_words:
                    if word in sent.text.lower():
                        # Check if it's a noun in this sentence
                        for token in sent:
                            if word in token.text.lower() and token.pos_ in ['NOUN', 'PROPN']:
                                relevance += 0.1
                                break
                        break
                if relevance >= 1.0:
                    break
            
            return min(1.0, relevance)
            
        except Exception as e:
            logger.warning(f"Semantic similarity scoring failed: {e}")
            return 0.0
    
    def _score_by_entities(self, article: Article, topic_entities: List[ExtractedEntity], topic_keywords: List[str]) -> Tuple[float, List[Dict]]:
        article_tags = self.database.get_article_entity_tags(article.id)
        matched = []
        entity_score = 0.0
        
        for topic_entity in topic_entities:
            for tag in article_tags:
                if tag['entity_text'].lower() == topic_entity.text.lower():
                    entity_score += 0.4
                    matched.append({'matched_text': tag['entity_text'], 'entity_type': tag['entity_type'], 'match_type': 'exact'})
                elif topic_entity.text.lower() in tag['entity_text'].lower():
                    entity_score += 0.2
                    matched.append({'matched_text': tag['entity_text'], 'entity_type': tag['entity_type'], 'match_type': 'partial'})
        
        keyword_score = 0.0
        article_text = f"{article.title} {article.content or ''} {article.summary or ''}".lower()
        for keyword in topic_keywords:
            if keyword.lower() in article.title.lower():
                keyword_score += 0.3
            elif keyword.lower() in article_text:
                keyword_score += 0.15
        
        return min(entity_score, 1.0) + min(keyword_score, 0.5), matched
    
    def _get_spacy_articles(self, topics: List[str], days: int, ai_only: bool, use_and_logic: bool) -> List[ScoredArticle]:
        if not self._spacy_available:
            return []
        
        cache_suffix = "_and" if use_and_logic else "_or"
        cached = self.cache.get(topics, days, cache_suffix)
        if cached:
            return [ScoredArticle(article=self._dict_to_article(i["article"]), confidence=i["confidence"],
                    matched_entities=i.get("matched_entities", [])) for i in cached.get("scored_articles", [])]
        
        articles = self.database.get_articles(limit=5000, ai_only=ai_only)
        
        # Filter by date - exclude articles without published_at
        cutoff_date = datetime.now() - timedelta(days=days)
        dated_articles = []
        for a in articles:
            if not a.published_at:
                logger.debug(f"Skipping article {a.id} (no publish date) for spacy analysis")
                continue
            
            article_date = a.published_at
            if article_date.tzinfo is not None:
                article_date = article_date.astimezone(None).replace(tzinfo=None)
            
            cutoff_naive = cutoff_date.replace(tzinfo=None)
            if article_date >= cutoff_naive:
                dated_articles.append(a)
        
        articles_dict = [self._article_to_dict(a) for a in dated_articles]
        filtered = self._filter_by_keywords(articles_dict, topics, use_and_logic)
        
        if not filtered:
            return []
        
        scored = self._score_with_spacy(filtered, topics, use_and_logic)
        cache_data = {
            "scored_articles": [
                {"article": self._article_to_dict(sa.article), "confidence": sa.confidence,
                 "matched_entities": sa.matched_entities} 
                for sa in scored
            ]
        }
        self.cache.set(topics, days, cache_data, suffix=cache_suffix)
        return scored
    
    def _filter_by_keywords(self, articles: List[Dict], topics: List[str], use_and_logic: bool) -> List[Dict]:
        filtered = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('content', '')} {article.get('summary', '')}".lower()
            if use_and_logic and all(t.lower() in text for t in topics):
                filtered.append(article)
            elif not use_and_logic and any(t.lower() in text for t in topics):
                filtered.append(article)
        return filtered
    
    def _score_with_spacy(self, articles: List[Dict], topics: List[str], use_and_logic: bool) -> List[ScoredArticle]:
        scored = []
        for article_dict in articles:
            if use_and_logic and len(topics) > 1:
                topic_scores = [self._score_article_for_topic(article_dict, t)[0] for t in topics]
                confidence = min(topic_scores) if topic_scores else 0.0
            else:
                confidence, _ = self._score_article_for_topic(article_dict, topics)
            
            if confidence >= self.CONFIDENCE_THRESHOLD:
                scored.append(ScoredArticle(article=self._dict_to_article(article_dict), confidence=confidence, matched_entities=[]))
        return scored
    
    def _score_article_for_topic(self, article: Dict, topic: Any) -> Tuple[float, Set[str]]:
        if not self._spacy_extractor:
            return 0.0, set()
        
        topic_str = " ".join(topic) if isinstance(topic, list) else str(topic)
        try:
            title_terms = self._spacy_extractor.extract_terms(article.get("title", ""), article.get("id"))
            content_terms = self._spacy_extractor.extract_terms(article.get("content", ""), article.get("id"))
            topic_terms = self._spacy_extractor.extract_terms(topic_str, None)
            
            matched = set()
            score = 0.0
            for term in topic_terms:
                for tt in title_terms:
                    if term.text.lower() == tt.text.lower():
                        score += 0.3
                        matched.add(term.text)
                for ct in content_terms:
                    if term.text.lower() == ct.text.lower():
                        score += 0.2
                        matched.add(term.text)
            return min(score, 1.0), matched
        except Exception as e:
            logger.warning(f"spaCy scoring failed: {e}")
            return 0.0, set()
    
    def _combine_scored_articles(self, entity_scored: List[ScoredArticle], spacy_scored: List[ScoredArticle]) -> List[ScoredArticle]:
        combined = []
        seen_ids = set()
        
        for sa in entity_scored:
            if sa.article.id not in seen_ids:
                combined.append(sa)
                seen_ids.add(sa.article.id)
        
        for sa in spacy_scored:
            if sa.article.id in seen_ids:
                for existing in combined:
                    if existing.article.id == sa.article.id:
                        existing.confidence = max(existing.confidence, sa.confidence)
                        break
            else:
                combined.append(sa)
                seen_ids.add(sa.article.id)
        
        combined.sort(key=lambda x: x.confidence, reverse=True)
        return combined
    
    def _generate_entity_insights(self, scored_articles: List[ScoredArticle]) -> Dict[str, Any]:
        insights = {'companies': {}, 'products': {}, 'technologies': {}, 'people': {}}
        for sa in scored_articles:
            for match in sa.matched_entities:
                etype = match.get('entity_type', '').lower()
                etext = match.get('matched_text', match.get('text', ''))
                if 'company' in etype:
                    insights['companies'][etext] = insights['companies'].get(etext, 0) + 1
                elif 'product' in etype:
                    insights['products'][etext] = insights['products'].get(etext, 0) + 1
        return insights
    
    def _group_by_entity(self, scored_articles: List[ScoredArticle], entity_type: str = 'company') -> Dict[str, List[ScoredArticle]]:
        grouped = {}
        for sa in scored_articles:
            for match in sa.matched_entities:
                if entity_type in match.get('entity_type', '').lower():
                    etext = match.get('matched_text', 'Unknown')
                    if etext not in grouped:
                        grouped[etext] = []
                    grouped[etext].append(sa)
                    break
        return grouped
    
    def _generate_digest_content(self, topics: List[str], days: int, scored_articles: List[ScoredArticle],
                                 insights: Dict[str, Any], grouped: Dict[str, List[ScoredArticle]]) -> str:
        topics_str = ", ".join(topics)
        digest = f"""# Topic Analysis: {topics_str} (Last {days} Days)

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*  
**Topics:** {topics_str}  
**Method:** Entity-aware + spaCy semantic analysis

"""
        digest += self._format_insights(insights)
        digest += "\n---\n\n"
        digest += self._format_grouped(grouped)
        digest += self._format_articles(scored_articles)
        digest += self._format_stats(scored_articles)
        return digest
    
    def _format_insights(self, insights: Dict[str, Any]) -> str:
        section = "## 📊 Entity Insights\n\n"
        if insights.get('companies'):
            section += "**Companies:**\n"
            for c, n in list(insights['companies'].items())[:10]:
                section += f"- {c} ({n})\n"
        if insights.get('products'):
            section += "\n**Products:**\n"
            for p, n in list(insights['products'].items())[:10]:
                section += f"- {p} ({n})\n"
        return section
    
    def _format_grouped(self, grouped: Dict[str, List[ScoredArticle]]) -> str:
        if not grouped:
            return ""
        section = "## 📰 Articles by Company\n\n"
        for company, articles in list(grouped.items())[:5]:
            section += f"### {company}\n\n"
            for sa in articles[:3]:
                section += f"- **{sa.article.title}** (confidence: {sa.confidence:.2f})\n"
                section += f"  {sa.article.url}\n\n"
        return section
    
    def _format_articles(self, scored_articles: List[ScoredArticle]) -> str:
        """Format articles, separating dated and undated articles."""
        section = "## 📋 All Articles\n\n"
        
        # Separate dated and undated articles
        dated_articles = []
        undated_articles = []
        
        for sa in scored_articles:
            # Check if this is marked as undated
            is_undated = any(
                isinstance(m, dict) and m.get('is_undated') 
                for m in sa.matched_entities
            )
            
            if is_undated:
                undated_articles.append(sa)
            else:
                dated_articles.append(sa)
        
        # Show dated articles first
        if dated_articles:
            for i, sa in enumerate(dated_articles[:20], 1):
                section += f"### {i}. {sa.article.title}\n\n"
                section += f"**Confidence:** {sa.confidence:.2f}  \n"
                if sa.article.summary:
                    section += f"**Summary:** {sa.article.summary}\n\n"
                section += f"**Source:** {sa.article.source_name}  \n"
                section += f"**URL:** {sa.article.url}  \n\n"
                section += "---\n\n"
        
        # Show undated articles in separate section
        if undated_articles:
            section += "## 📅 Undated Articles\n\n"
            section += "*Articles without publication dates (may be older or newer than specified range)*\n\n"
            
            for i, sa in enumerate(undated_articles[:10], 1):
                section += f"### {i}. {sa.article.title}\n\n"
                section += f"**Confidence:** {sa.confidence:.2f}  \n"
                section += f"**Source:** {sa.article.source_name}  \n"
                section += f"**URL:** {sa.article.url}  \n\n"
                section += "---\n\n"
        
        return section
    
    def _format_stats(self, scored_articles: List[ScoredArticle]) -> str:
        section = "## 📈 Statistics\n\n"
        
        # Count dated vs undated
        dated_count = sum(
            1 for sa in scored_articles 
            if not any(isinstance(m, dict) and m.get('is_undated') for m in sa.matched_entities)
        )
        undated_count = len(scored_articles) - dated_count
        
        section += f"**Total:** {len(scored_articles)} articles\n"
        section += f"- Dated articles: {dated_count}\n"
        section += f"- Undated articles: {undated_count}\n\n"
        
        if scored_articles:
            avg = sum(sa.confidence for sa in scored_articles) / len(scored_articles)
            section += f"**Avg confidence:** {avg:.2f}\n"
        
        return section
    
    def _generate_empty_digest(self, topics: List[str], days: int) -> str:
        return f"""# Topic Analysis: {", ".join(topics)} (Last {days} Days)

*No articles found.*
"""
    
    def _article_to_dict(self, article: Article) -> Dict:
        """Convert article to dict, handling datetime serialization."""
        return {
            'id': article.id, 'title': article.title, 'content': article.content,
            'summary': article.summary, 'url': article.url, 'source_name': article.source_name,
            'author': article.author, 
            'published_at': article.published_at.isoformat() if article.published_at else None,
            'category': article.category
        }
    
    def _dict_to_article(self, d: Dict) -> Article:
        """Convert dict back to article, handling datetime deserialization."""
        from datetime import datetime
        
        # Parse published_at if it's a string
        published_at = d.get('published_at')
        if isinstance(published_at, str):
            try:
                published_at = datetime.fromisoformat(published_at)
            except:
                published_at = None
        
        return Article(id=d.get('id'), title=d.get('title'), content=d.get('content'),
                       summary=d.get('summary'), url=d.get('url'), source_name=d.get('source_name'),
                       author=d.get('author'), published_at=published_at, category=d.get('category'))


def create_unified_digest_generator(database: Database, cache_ttl_hours: int = 6) -> UnifiedDigestGenerator:
    """Factory function to create unified digest generator."""
    return UnifiedDigestGenerator(database, cache_ttl_hours)
