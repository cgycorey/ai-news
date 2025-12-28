#!/usr/bin/env python3
"""Rescore all articles in database with updated confidence scorer."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_news.database import Database
from ai_news.confidence_scorer import ConfidenceScorer


def rescore_all_articles(db_path: str = "data/production/ai_news.db"):
    """Rescore all articles with updated confidence scorer."""
    print("=" * 70)
    print("RESCORING ALL ARTICLES")
    print("=" * 70)

    db = Database(db_path)
    scorer = ConfidenceScorer(db)
    scorer.refresh_learned_keywords()

    # Get all articles directly from database
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT id, title, content, summary, url, author, published_at, source_name, category, region FROM articles")
    rows = cursor.fetchall()
    conn.close()

    # Convert to Article objects
    from ai_news.database import Article
    articles = []
    for row in rows:
        articles.append(Article(
            id=row[0],
            title=row[1],
            content=row[2],
            summary=row[3],
            url=row[4],
            author=row[5],
            published_at=None,  # Skip date parsing for rescore
            source_name=row[7],
            category=row[8],
            region=row[9]
        ))

    print(f"\nFound {len(articles)} articles to rescore...")

    updated = 0
    rejected = 0
    high_confidence = 0

    for article in articles:
        # Calculate new confidence
        old_confidence = article.ai_confidence
        new_confidence = scorer.calculate_confidence(article)

        # Update article
        if new_confidence != old_confidence:
            article.ai_confidence = new_confidence
            article.ai_review_status = scorer.get_review_status(new_confidence)
            article.ai_relevant = (new_confidence >= 0.7)

            # Save to database (update)
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE articles
                SET ai_confidence = ?, ai_review_status = ?, ai_relevant = ?
                WHERE id = ?
            """, (new_confidence, article.ai_review_status, article.ai_relevant, article.id))
            conn.commit()
            conn.close()

            updated += 1

            if new_confidence < 0.5:
                rejected += 1
            elif new_confidence >= 0.7:
                high_confidence += 1

            print(f"  Updated: {article.title[:50]}... | {old_confidence:.2f} → {new_confidence:.2f}")

    print("\n" + "=" * 70)
    print("RESCORING COMPLETE")
    print("=" * 70)
    print(f"Total articles: {len(articles)}")
    print(f"Updated: {updated}")
    print(f"Now rejected (<0.5): {rejected}")
    print(f"Now high confidence (≥0.7): {high_confidence}")
    print()


if __name__ == "__main__":
    rescore_all_articles()
