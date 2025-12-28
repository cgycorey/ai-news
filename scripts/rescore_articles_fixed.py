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
    print("RESCORING ALL ARTICLES WITH FIXED SCORER")
    print("=" * 70)

    db = Database(db_path)
    scorer = ConfidenceScorer(db)
    scorer.refresh_learned_keywords()

    # Get all articles using database method
    articles = db.get_articles(ai_only=False, limit=1000)
    print(f"\n📊 Found {len(articles)} articles to rescore...")

    updated = 0
    rejected_count = 0
    accepted_count = 0
    false_positives = []

    for i, article in enumerate(articles):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(articles)}...")

        # Calculate new confidence
        old_confidence = article.ai_confidence
        old_relevant = article.ai_relevant
        new_confidence = scorer.calculate_confidence(article)

        # Check if changed
        if abs(new_confidence - old_confidence) > 0.01:
            article.ai_confidence = new_confidence
            article.ai_review_status = scorer.get_review_status(new_confidence)
            article.ai_relevant = (new_confidence >= 0.7)

            # Update in database
            import sqlite3
            conn = sqlite3.connect(db_path, timeout=30)
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    UPDATE articles
                    SET ai_confidence = ?, ai_review_status = ?, ai_relevant = ?
                    WHERE id = ?
                """, (new_confidence, article.ai_review_status, article.ai_relevant, article.id))
                conn.commit()
            finally:
                conn.close()

            updated += 1

            # Track changes
            if old_relevant and not article.ai_relevant:
                false_positives.append((article.title, old_confidence, new_confidence))
                rejected_count += 1
            elif not old_relevant and article.ai_relevant:
                accepted_count += 1

    print("\n" + "=" * 70)
    print("✅ RESCORING COMPLETE")
    print("=" * 70)
    print(f"📊 Total articles: {len(articles)}")
    print(f"🔄 Updated: {updated}")
    print(f"❌ Now rejected (<0.7): {rejected_count}")
    print(f"✓ Now accepted (≥0.7): {accepted_count}")

    if false_positives:
        print(f"\n🎯 FALSE POSITIVES FIXED (sample):")
        for title, old, new in false_positives[:10]:
            print(f"  • {title[:60]}... | {old:.2f} → {new:.2f}")

    print()


if __name__ == "__main__":
    rescore_all_articles()
