#!/usr/bin/env python3
"""Tag all existing articles with entities.

This script batch-processes all articles in the database and extracts entities for them.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_news.database import Database
from ai_news.article_tagger import get_article_tagger
from ai_news.config import Config


def tag_all_articles(limit=None, batch_size=100):
    """Tag all articles in the database with entities.
    
    Args:
        limit: Maximum number of articles to process (None = all)
        batch_size: Number of articles to process before committing
    """
    # Load config to get database path
    config = Config()
    db_path = config.database_path
    
    print(f"📦 Database: {db_path}")
    database = Database(db_path)
    
    # Get all articles
    print("📖 Loading articles from database...")
    all_articles = database.get_articles(limit=limit or 10000)
    total = len(all_articles)
    
    if total == 0:
        print("❌ No articles found in database")
        return
    
    print(f"✅ Found {total} articles")
    print()
    
    # Get tagger
    tagger = get_article_tagger()
    
    # Process articles
    tagged = 0
    skipped = 0
    errors = 0
    
    for i, article in enumerate(all_articles, 1):
        try:
            # Check if already tagged
            if article.id is None:
                skipped += 1
                continue
            
            existing_tags = database.get_article_entity_tags(article.id)
            if existing_tags:
                skipped += 1
                if i % 50 == 0:
                    print(f"  Progress: {i}/{total} | Tagged: {tagged} | Skipped: {skipped} | Errors: {errors}")
                continue
            
            # Tag the article
            tags = tagger.tag_article(article)
            
            if tags:
                # Save tags
                tagger.save_tags(article.id, tags, database)
                tagged += 1
                
                if tagged % 10 == 0:
                    print(f"  Progress: {i}/{total} | Tagged: {tagged} | Skipped: {skipped} | Errors: {errors}")
            else:
                skipped += 1
            
            # Batch commit
            if i % batch_size == 0:
                print(f"\n  📊 Batch complete at {i}/{total}")
                print(f"     Tagged: {tagged} | Skipped: {skipped} | Errors: {errors}\n")
        
        except Exception as e:
            errors += 1
            print(f"\n  ❌ Error tagging article {article.id}: {e}\n")
    
    # Summary
    print("\n" + "="*60)
    print("TAGGING COMPLETE")
    print("="*60)
    print(f"Total articles:     {total}")
    print(f"Successfully tagged: {tagged}")
    print(f"Skipped (no tags):  {skipped}")
    print(f"Errors:             {errors}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tag all articles with entities")
    parser.add_argument("--limit", type=int, help="Limit number of articles to process")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch commit size")
    
    args = parser.parse_args()
    
    print("="*60)
    print("BATCH ARTICLE TAGGING")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tag_all_articles(limit=args.limit, batch_size=args.batch_size)
    
    print()
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
