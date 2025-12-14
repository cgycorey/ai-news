"""Intelligence layer database operations."""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import sqlite3
from pathlib import Path

from .database import (
    Entity, Topic, EntityMention
)


class IntelligenceDB:
    """Database operations for the intelligence layer."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
    
    def _serialize_list(self, data: Optional[List]) -> Optional[str]:
        """Serialize list to JSON string."""
        return json.dumps(data) if data else None
    
    def _deserialize_list(self, data: Optional[str]) -> List:
        """Deserialize JSON string to list."""
        return json.loads(data) if data else []
    
    def _serialize_dict(self, data: Optional[Dict]) -> Optional[str]:
        """Serialize dict to JSON string."""
        return json.dumps(data) if data else None
    
    def _deserialize_dict(self, data: Optional[str]) -> Dict:
        """Deserialize JSON string to dict."""
        return json.loads(data) if data else {}

    # === ENTITY OPERATIONS ===
    
    def create_entity(self, entity: Entity) -> Optional[int]:
        """Create a new entity."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO entities 
                    (name, entity_type, description, aliases, metadata, confidence_score, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity.name,
                    entity.entity_type,
                    entity.description,
                    self._serialize_list(entity.aliases),
                    self._serialize_dict(entity.metadata),
                    entity.confidence_score,
                    entity.created_at or datetime.now(),
                    entity.updated_at or datetime.now()
                ))
                return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error creating entity: {e}")
            return None
    
    def get_entity(self, entity_id: int) -> Optional[Entity]:
        """Get entity by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
            
            if row:
                return Entity(
                    id=row['id'],
                    name=row['name'],
                    entity_type=row['entity_type'],
                    description=row['description'],
                    aliases=self._deserialize_list(row['aliases']),
                    metadata=self._deserialize_dict(row['metadata']),
                    confidence_score=row['confidence_score'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
                )
        return None
    
    def get_entity_by_name(self, name: str, entity_type: Optional[str] = None) -> Optional[Entity]:
        """Get entity by name and optional type."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if entity_type:
                row = conn.execute(
                    "SELECT * FROM entities WHERE name = ? AND entity_type = ?", 
                    (name, entity_type)
                ).fetchone()
            else:
                row = conn.execute("SELECT * FROM entities WHERE name = ?", (name,)).fetchone()
            
            if row:
                return Entity(
                    id=row['id'],
                    name=row['name'],
                    entity_type=row['entity_type'],
                    description=row['description'],
                    aliases=self._deserialize_list(row['aliases']),
                    metadata=self._deserialize_dict(row['metadata']),
                    confidence_score=row['confidence_score'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
                )
        return None
    
    def get_entities(self, limit: int = 100, offset: int = 0, 
                      entity_type: Optional[str] = None) -> List[Entity]:
        """Get multiple entities with optional filtering and pagination."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            sql = "SELECT * FROM entities"
            params = []
            
            if entity_type:
                sql += " WHERE entity_type = ?"
                params.append(entity_type)
            
            sql += " ORDER BY confidence_score DESC, name ASC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            rows = conn.execute(sql, params).fetchall()
            
            return [
                Entity(
                    id=row['id'],
                    name=row['name'],
                    entity_type=row['entity_type'],
                    description=row['description'],
                    aliases=self._deserialize_list(row['aliases']),
                    metadata=self._deserialize_dict(row['metadata']),
                    confidence_score=row['confidence_score'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
                )
                for row in rows
            ]
    
    def search_entities(self, query: str, entity_type: Optional[str] = None, 
                       limit: int = 50) -> List[Entity]:
        """Search entities by name or description."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            sql = """
                SELECT * FROM entities 
                WHERE (name LIKE ? OR description LIKE ?)
            """
            params = [f"%{query}%", f"%{query}%"]
            
            if entity_type:
                sql += " AND entity_type = ?"
                params.append(entity_type)
            
            sql += " ORDER BY confidence_score DESC, name ASC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(sql, params).fetchall()
            
            return [
                Entity(
                    id=row['id'],
                    name=row['name'],
                    entity_type=row['entity_type'],
                    description=row['description'],
                    aliases=self._deserialize_list(row['aliases']),
                    metadata=self._deserialize_dict(row['metadata']),
                    confidence_score=row['confidence_score'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
                )
                for row in rows
            ]
    
    def get_entities_by_type(self, entity_type: str, limit: Optional[int] = None) -> List[Entity]:
        """Get all entities of a specific type."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            sql = "SELECT * FROM entities WHERE entity_type = ? ORDER BY confidence_score DESC, name ASC"
            params = [entity_type]
            
            if limit:
                sql += " LIMIT ?"
                params.append(limit)
            
            rows = conn.execute(sql, params).fetchall()
            
            return [
                Entity(
                    id=row['id'],
                    name=row['name'],
                    entity_type=row['entity_type'],
                    description=row['description'],
                    aliases=self._deserialize_list(row['aliases']),
                    metadata=self._deserialize_dict(row['metadata']),
                    confidence_score=row['confidence_score'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
                )
                for row in rows
            ]
    
    def update_entity(self, entity: Entity) -> bool:
        """Update an existing entity."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE entities 
                    SET name = ?, entity_type = ?, description = ?, aliases = ?, 
                        metadata = ?, confidence_score = ?, updated_at = ?
                    WHERE id = ?
                """, (
                    entity.name,
                    entity.entity_type,
                    entity.description,
                    self._serialize_list(entity.aliases),
                    self._serialize_dict(entity.metadata),
                    entity.confidence_score,
                    datetime.now(),
                    entity.id
                ))
                return conn.total_changes > 0
        except sqlite3.Error as e:
            print(f"Error updating entity: {e}")
            return False
    
    def delete_entity(self, entity_id: int) -> bool:
        """Delete an entity."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
                return conn.total_changes > 0
        except sqlite3.Error as e:
            print(f"Error deleting entity: {e}")
            return False
    
    def get_top_entities_by_mentions(self, limit: int = 20, 
                                   entity_type: Optional[str] = None,
                                   days_back: Optional[int] = None) -> List[Tuple[Entity, int]]:
        """Get top entities by mention count."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            sql = """
                SELECT e.*, COUNT(em.id) as mention_count
                FROM entities e
                LEFT JOIN entity_mentions em ON e.id = em.entity_id
            """
            params = []
            
            if entity_type:
                sql += " WHERE e.entity_type = ?"
                params.append(entity_type)
            
            if days_back:
                if entity_type:
                    sql += " AND"
                else:
                    sql += " WHERE"
                sql += " em.created_at >= datetime('now', '-{} days')".format(days_back)
            
            sql += " GROUP BY e.id ORDER BY mention_count DESC, e.name ASC LIMIT ?"
            params.append(limit)
            
            rows = conn.execute(sql, params).fetchall()
            
            results = []
            for row in rows:
                entity = Entity(
                    id=row['id'],
                    name=row['name'],
                    entity_type=row['entity_type'],
                    description=row['description'],
                    aliases=self._deserialize_list(row['aliases']),
                    metadata=self._deserialize_dict(row['metadata']),
                    confidence_score=row['confidence_score'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
                )
                results.append((entity, row['mention_count'] or 0))
            
            return results

    # === TOPIC OPERATIONS ===
    
    def create_topic(self, topic: Topic) -> Optional[int]:
        """Create a new topic."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO topics 
                    (name, description, keywords, topic_cluster_id, weight, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    topic.name,
                    topic.description,
                    self._serialize_list(topic.keywords),
                    topic.topic_cluster_id,
                    topic.weight,
                    topic.created_at or datetime.now(),
                    topic.updated_at or datetime.now()
                ))
                return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error creating topic: {e}")
            return None
    
    def get_topic(self, topic_id: int) -> Optional[Topic]:
        """Get topic by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
            
            if row:
                return Topic(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    keywords=self._deserialize_list(row['keywords']),
                    topic_cluster_id=row['topic_cluster_id'],
                    weight=row['weight'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
                )
        return None
    
    def get_trending_topics(self, limit: int = 20) -> List[Topic]:
        """Get trending topics ordered by weight."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM topics ORDER BY weight DESC, name ASC LIMIT ?",
                (limit,)
            ).fetchall()
            
            return [
                Topic(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    keywords=self._deserialize_list(row['keywords']),
                    topic_cluster_id=row['topic_cluster_id'],
                    weight=row['weight'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
                )
                for row in rows
            ]
    
    def update_topic(self, topic: Topic) -> bool:
        """Update an existing topic."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE topics 
                    SET name = ?, description = ?, keywords = ?, topic_cluster_id = ?,
                        weight = ?, updated_at = ?
                    WHERE id = ?
                """, (
                    topic.name,
                    topic.description,
                    self._serialize_list(topic.keywords),
                    topic.topic_cluster_id,
                    topic.weight,
                    datetime.now(),
                    topic.id
                ))
                return conn.total_changes > 0
        except sqlite3.Error as e:
            print(f"Error updating topic: {e}")
            return False

    # === ARTICLE-ENTITY RELATIONSHIPS ===
    
    def link_article_entity(self, article_id: int, entity_id: int, 
                           relevance_score: float = 1.0,
                           mention_positions: Optional[List[int]] = None) -> bool:
        """Link an article to an entity."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO article_entities 
                    (article_id, entity_id, relevance_score, mention_positions, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    article_id,
                    entity_id,
                    relevance_score,
                    self._serialize_list(mention_positions),
                    datetime.now()
                ))
                return True
        except sqlite3.Error as e:
            print(f"Error linking article to entity: {e}")
            return False
    
    def get_article_entities(self, article_id: int) -> List[Tuple[Entity, float]]:
        """Get all entities linked to an article with relevance scores."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT e.*, ae.relevance_score
                FROM entities e
                JOIN article_entities ae ON e.id = ae.entity_id
                WHERE ae.article_id = ?
                ORDER BY ae.relevance_score DESC
            """, (article_id,)).fetchall()
            
            return [
                (
                    Entity(
                        id=row['id'],
                        name=row['name'],
                        entity_type=row['entity_type'],
                        description=row['description'],
                        aliases=self._deserialize_list(row['aliases']),
                        metadata=self._deserialize_dict(row['metadata']),
                        confidence_score=row['confidence_score'],
                        created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                        updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
                    ),
                    row['relevance_score']
                )
                for row in rows
            ]
    
    def get_entity_articles(self, entity_id: int, limit: int = 50) -> List[Tuple[int, float]]:
        """Get articles linked to an entity with relevance scores."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT article_id, relevance_score
                FROM article_entities
                WHERE entity_id = ?
                ORDER BY relevance_score DESC
                LIMIT ?
            """, (entity_id, limit)).fetchall()
            
            return [(row[0], row[1]) for row in rows]

    # === ENTITY MENTIONS ===
    
    def create_entity_mention(self, mention: EntityMention) -> Optional[int]:
        """Create a new entity mention."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO entity_mentions 
                    (article_id, entity_id, mention_count, sentiment_score, context_snippets,
                     confidence_score, mention_positions, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    mention.article_id,
                    mention.entity_id,
                    mention.mention_count,
                    mention.sentiment_score,
                    self._serialize_list(mention.context_snippets),
                    mention.confidence_score,
                    self._serialize_list(mention.mention_positions),
                    mention.created_at or datetime.now()
                ))
                return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error creating entity mention: {e}")
            return None
    
    def get_entity_mentions(self, entity_id: int, limit: int = 50) -> List[EntityMention]:
        """Get mentions of a specific entity."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM entity_mentions 
                WHERE entity_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (entity_id, limit)).fetchall()
            
            return [
                EntityMention(
                    id=row['id'],
                    article_id=row['article_id'],
                    entity_id=row['entity_id'],
                    mention_count=row['mention_count'],
                    sentiment_score=row['sentiment_score'],
                    context_snippets=self._deserialize_list(row['context_snippets']),
                    confidence_score=row['confidence_score'],
                    mention_positions=self._deserialize_list(row['mention_positions']),
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
                )
                for row in rows
            ]
    
    def get_entity_sentiment_trend(self, entity_id: int, days_back: int = 30) -> Dict[str, Any]:
        """Get sentiment trend for an entity over time."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT 
                    DATE(created_at) as date,
                    AVG(sentiment_score) as avg_sentiment,
                    COUNT(*) as mention_count
                FROM entity_mentions 
                WHERE entity_id = ? 
                    AND created_at >= datetime('now', '-{} days')
                    AND sentiment_score IS NOT NULL
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """.format(days_back), (entity_id,)).fetchall()
            
            return {
                "daily_data": [
                    {
                        "date": row[0],
                        "avg_sentiment": row[1],
                        "mention_count": row[2]
                    }
                    for row in rows
                ],
                "overall_sentiment": sum(row[1] * row[2] for row in rows) / sum(row[2] for row in rows) if rows else 0.0,
                "total_mentions": sum(row[2] for row in rows)
            }

    # === REMOVED ACADEMIC FEATURES ===
    # ProductIdea and CompetitiveAnalysis classes removed for business focus
    # Keeping practical business intelligence: entities, topics, mentions

    # === BULK OPERATIONS ===
    
    def bulk_create_entities(self, entities: List[Entity]) -> List[int]:
        """Create multiple entities efficiently."""
        created_ids = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for entity in entities:
                    cursor = conn.execute("""
                        INSERT OR IGNORE INTO entities 
                        (name, entity_type, description, aliases, metadata, confidence_score, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entity.name,
                        entity.entity_type,
                        entity.description,
                        self._serialize_list(entity.aliases),
                        self._serialize_dict(entity.metadata),
                        entity.confidence_score,
                        entity.created_at or datetime.now(),
                        entity.updated_at or datetime.now()
                    ))
                    if cursor.lastrowid:
                        created_ids.append(cursor.lastrowid)
        except sqlite3.Error as e:
            print(f"Error in bulk entity creation: {e}")
        
        return created_ids
    
    def bulk_link_article_entities(self, links: List[Tuple[int, int, float, List[int]]]) -> int:
        """Bulk link articles to entities.
        
        Args:
            links: List of (article_id, entity_id, relevance_score, mention_positions)
        
        Returns:
            Number of successful links created
        """
        created_count = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for article_id, entity_id, relevance_score, mention_positions in links:
                    conn.execute("""
                        INSERT OR REPLACE INTO article_entities 
                        (article_id, entity_id, relevance_score, mention_positions, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        article_id,
                        entity_id,
                        relevance_score,
                        self._serialize_list(mention_positions),
                        datetime.now()
                    ))
                    created_count += 1
        except sqlite3.Error as e:
            print(f"Error in bulk linking article entities: {e}")
        
        return created_count

    # === ANALYTICS AND AGGREGATION ===
    
    def get_intelligence_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the intelligence layer."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Entity stats
            stats['entities'] = {}
            for entity_type in ['company', 'product', 'technology', 'person']:
                count = conn.execute(
                    "SELECT COUNT(*) FROM entities WHERE entity_type = ?",
                    (entity_type,)
                ).fetchone()[0]
                stats['entities'][entity_type] = count
            stats['entities']['total'] = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            
            # Topic stats
            stats['topics'] = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
            
            # Relationship stats
            stats['article_entity_links'] = conn.execute("SELECT COUNT(*) FROM article_entities").fetchone()[0]
            stats['entity_mentions'] = conn.execute("SELECT COUNT(*) FROM entity_mentions").fetchone()[0]
            
            # Academic features removed - focusing on business intelligence
            
            # Coverage stats
            articles_with_entities = conn.execute("""
                SELECT COUNT(DISTINCT article_id) FROM article_entities
            """).fetchone()[0]
            
            total_articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            stats['coverage'] = {
                'articles_with_entities': articles_with_entities,
                'total_articles': total_articles,
                'coverage_percentage': (articles_with_entities / total_articles * 100) if total_articles > 0 else 0
            }
            
            return stats
    
    def get_entity_co_occurrence_network(self, limit: int = 50) -> List[Tuple[int, int, int]]:
        """Get entity co-occurrence relationships.
        
        Returns:
            List of (entity1_id, entity2_id, co_occurrence_count)
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT 
                    ae1.entity_id as entity1_id,
                    ae2.entity_id as entity2_id,
                    COUNT(*) as co_occurrence_count
                FROM article_entities ae1
                JOIN article_entities ae2 ON ae1.article_id = ae2.article_id
                WHERE ae1.entity_id < ae2.entity_id
                GROUP BY ae1.entity_id, ae2.entity_id
                HAVING co_occurrence_count >= 2
                ORDER BY co_occurrence_count DESC
                LIMIT ?
            """, (limit,)).fetchall()
            
            return [(row[0], row[1], row[2]) for row in rows]
    
    def get_trending_entities_by_topic(self, topic_id: int, days_back: int = 7) -> List[Tuple[Entity, int]]:
        """Get trending entities within a specific topic."""
        # This is a placeholder - would need topic-entity relationships for full implementation
        return []