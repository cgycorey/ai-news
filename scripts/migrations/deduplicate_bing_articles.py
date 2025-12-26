#!/usr/bin/env python3
"""Deduplicate Bing News articles with tracking URLs.

This script identifies and removes duplicate articles caused by Bing News
tracking URLs (different tid= parameters pointing to the same article).
"""

import sys
import os
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
from collections import defaultdict

# Add parent directory to path
script_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(script_dir))

from src.ai_news.database import Database


def extract_canonical_url(url: str) -> str:
    """Extract canonical URL from Bing News tracking URL."""
    if not url:
        return url

    try:
        parsed = urlparse(url)

        # Handle Bing News redirects
        if 'apiclick.aspx' in parsed.path:
            params = parse_qs(parsed.query)
            canonical = params.get('url', [url])[0]
            return unquote(canonical)

        return url
    except:
        return url


def deduplicate_database(db_path: str, dry_run: bool = True):
    """Remove duplicate articles from database.

    Args:
        db_path: Path to database
        dry_run: If True, only report what would be deleted
    """
    db = Database(db_path)

    print("🔍 Scanning database for duplicates...\n")

    # Get all articles (batch processing to avoid memory issues)
    all_articles = []
    offset = 0
    batch_size = 1000

    while True:
        batch = db.get_articles(limit=batch_size)
        if not batch:
            break
        all_articles.extend(batch)
        if len(batch) < batch_size:
            break

    articles = all_articles

    print(f"📊 Total articles in database: {len(articles)}\n")

    # Group by canonical URL
    canonical_groups = defaultdict(list)
    for article in articles:
        canonical = extract_canonical_url(article.url)
        canonical_groups[canonical].append(article)

    # Find duplicates
    duplicates_to_delete = []
    kept_articles = []
    duplicate_groups = []

    for canonical, group in canonical_groups.items():
        if len(group) > 1:
            # Sort by ID (keep oldest)
            group.sort(key=lambda a: a.id)

            # Keep the first one, mark others for deletion
            kept_articles.append(group[0])
            duplicates_to_delete.extend(group[1:])

            duplicate_groups.append({
                'canonical': canonical,
                'keep': group[0],
                'delete': group[1:]
            })
        else:
            kept_articles.append(group[0])

    # Print summary
    print("=" * 80)
    print("DUPLICATE DETECTION SUMMARY")
    print("=" * 80)
    print(f"Total articles:           {len(articles):,}")
    print(f"Unique articles:          {len(kept_articles):,}")
    print(f"Duplicates found:         {len(duplicates_to_delete):,}")
    print(f"Duplicate percentage:     {len(duplicates_to_delete)/len(articles)*100:.1f}%")
    print()

    # Show first 10 duplicate groups
    if duplicate_groups:
        print("Sample duplicate groups (first 10):")
        print("-" * 80)

        for i, group in enumerate(duplicate_groups[:10], 1):
            print(f"\n{i}. Canonical: {group['canonical'][:80]}...")
            print(f"   ✅ Keep:  ID {group['keep'].id} - {group['keep'].title[:60]}...")
            for dup in group['delete'][:3]:  # Show max 3 duplicates per group
                print(f"   ❌ Delete: ID {dup.id} - {dup.title[:60]}...")
            if len(group['delete']) > 3:
                print(f"   ... and {len(group['delete']) - 3} more duplicates")

    print()
    print("=" * 80)

    if dry_run:
        print("🔍 DRY RUN MODE - No changes made")
        print("To actually delete duplicates, run with dry_run=False")
        print()
        print("Usage:")
        print("  python scripts/migrations/deduplicate_bing_articles.py data/production/ai_news.db --delete")
    else:
        # Confirm before deleting
        print(f"⚠️  WARNING: About to DELETE {len(duplicates_to_delete):,} duplicate articles")
        print("This action cannot be undone!")
        print()
        response = input("Type 'yes' to confirm: ")

        if response.lower() != 'yes':
            print("❌ Cancelled")
            return

        # Delete duplicates
        print(f"\n🗑️  Deleting {len(duplicates_to_delete):,} duplicate articles...")

        # Use SQL to delete directly (more efficient)
        import sqlite3
        conn = sqlite3.connect(db_path)

        deleted_count = 0
        for article in duplicates_to_delete:
            try:
                conn.execute("DELETE FROM articles WHERE id = ?", (article.id,))
                deleted_count += 1
            except Exception as e:
                print(f"   ⚠️  Failed to delete article {article.id}: {e}")

        conn.commit()
        conn.close()

        print(f"\n✅ Successfully deleted {deleted_count:,} duplicate articles")
        print(f"Database now has {len(kept_articles):,} unique articles")

    print("=" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Deduplicate Bing News articles with tracking URLs'
    )
    parser.add_argument(
        'db_path',
        nargs='?',
        default='data/production/ai_news.db',
        help='Path to database file'
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Actually delete duplicates (default: dry-run mode)'
    )

    args = parser.parse_args()

    dry_run = not args.delete
    deduplicate_database(args.db_path, dry_run=dry_run)
