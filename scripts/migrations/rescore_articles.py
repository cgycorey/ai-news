#!/usr/bin/env python3
"""Re-score all existing articles with new confidence system."""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_news.database import Database
from ai_news.confidence_scorer import ConfidenceScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'data/production/ai_news.db'
    database = Database(db_path)

    scorer = ConfidenceScorer(database)

    # Get all articles
    articles = database.get_articles(limit=10000)
    logger.info(f"Re-scoring {len(articles)} articles...")

    updated = 0
    for article in articles:
        # Calculate new confidence
        confidence = scorer.calculate_confidence(article)
        review_status = scorer.get_review_status(confidence)
        ai_relevant = (confidence >= 0.7)

        # Update in database
        try:
            import sqlite3
            with sqlite3.connect(database.db_path) as conn:
                conn.execute("""
                    UPDATE articles
                    SET ai_confidence = ?,
                        ai_review_status = ?,
                        ai_relevant = ?
                    WHERE id = ?
                """, (confidence, review_status, int(ai_relevant), article.id))
                conn.commit()
                updated += 1
        except Exception as e:
            logger.error(f"Failed to update article {article.id}: {e}")

    logger.info(f"Updated {updated} articles")

    # Show stats
    with sqlite3.connect(database.db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        ai_relevant = conn.execute("SELECT COUNT(*) FROM articles WHERE ai_confidence >= 0.7").fetchone()[0]
        review_needed = conn.execute("SELECT COUNT(*) FROM articles WHERE ai_review_status = 'reviewed'").fetchone()[0]

        print(f"\n=== Stats ===")
        print(f"Total articles: {total}")
        print(f"AI-relevant (>=0.7): {ai_relevant} ({100*ai_relevant/total:.1f}%)")
        print(f"Need review (0.5-0.69): {review_needed} ({100*review_needed/total:.1f}%)")


if __name__ == '__main__':
    main()
