#!/usr/bin/env python3
"""Rescore all articles - fixes database lock with batched updates."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_news.database import Database
from ai_news.confidence_scorer import ConfidenceScorer
import sqlite3


def rescore_all_articles(db_path: str = "data/production/ai_news.db"):
    """Rescore all articles with batched updates to avoid locks."""
    print("=" * 70)
    print("RESCORING ALL ARTICLES - FIXING FALSE POSITIVES")
    print("=" * 70)

    db = Database(db_path)
    scorer = ConfidenceScorer(db)
    scorer.refresh_learned_keywords()

    # Get all articles
    articles = db.get_articles(ai_only=False, limit=10000)
    print(f"\n📊 Found {len(articles)} articles to rescore...")

    # Batch updates
    batch_size = 50
    updated_count = 0
    rejected_count = 0
    false_positives = []

    for i in range(0, len(articles), batch_size):
        batch = articles[i:i+batch_size]
        updates = []

        for article in batch:
            # Calculate new confidence
            old_confidence = article.ai_confidence
            old_relevant = article.ai_relevant
            new_confidence = scorer.calculate_confidence(article)

            # Check if changed
            if abs(new_confidence - old_confidence) > 0.01:
                new_relevant = (new_confidence >= 0.7)
                updates.append((
                    new_confidence,
                    'auto' if new_relevant else ('rejected' if new_confidence < 0.5 else 'reviewed'),
                    new_relevant,
                    article.id
                ))

                # Track false positives fixed
                if old_relevant and not new_relevant:
                    false_positives.append((article.title, old_confidence, new_confidence))
                    rejected_count += 1

        # Batch update
        if updates:
            try:
                conn = sqlite3.connect(db_path, timeout=30)
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()
                cursor.executemany("""
                    UPDATE articles
                    SET ai_confidence = ?, ai_review_status = ?, ai_relevant = ?
                    WHERE id = ?
                """, updates)
                conn.commit()
                conn.close()
                updated_count += len(updates)
            except Exception as e:
                print(f"  ⚠️  Batch update error: {e}")
                continue

        if (i + batch_size) % 100 == 0:
            print(f"  Progress: {min(i + batch_size, len(articles))}/{len(articles)}...")

    print("\n" + "=" * 70)
    print("✅ RESCORING COMPLETE")
    print("=" * 70)
    print(f"📊 Total articles: {len(articles)}")
    print(f"🔄 Updated: {updated_count}")
    print(f"❌ False positives fixed: {rejected_count}")

    if false_positives:
        print(f"\n🎯 FALSE POSITIVES FIXED:")
        for title, old, new in false_positives[:15]:
            print(f"  ❌→✓ {title[:60]}... | {old:.2f} → {new:.2f}")

    # Show new stats
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM articles WHERE ai_relevant = 1")
    ai_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM articles")
    total_count = cursor.fetchone()[0]
    conn.close()

    print(f"\n📈 New statistics:")
    print(f"   Total articles: {total_count}")
    print(f"   AI-relevant: {ai_count} ({100*ai_count/total_count:.1f}%)")
    print()


if __name__ == "__main__":
    rescore_all_articles()
