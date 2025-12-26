#!/usr/bin/env python
"""
Evaluate digest quality by comparing websearch vs RSS articles.

Metrics:
- AI relevance rate
- Content quality (length, completeness)
- Source diversity
- Language distribution
- Topic coverage
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ai_news.database import Database
import sqlite3
from datetime import datetime, timedelta
from collections import Counter


def evaluate_quality():
    """Compare websearch vs RSS article quality."""

    db = Database('data/production/ai_news.db')
    conn = sqlite3.connect('data/production/ai_news.db')
    conn.row_factory = sqlite3.Row

    print('='*70)
    print('DIGEST QUALITY EVALUATION')
    print('='*70)
    print()

    # Overall stats
    cursor = conn.execute('SELECT COUNT(*) FROM articles')
    total = cursor.fetchone()[0]

    cursor = conn.execute('SELECT COUNT(*) FROM articles WHERE url LIKE "%bing.com%"')
    websearch_count = cursor.fetchone()[0]

    cursor = conn.execute('SELECT COUNT(*) FROM articles WHERE url NOT LIKE "%bing.com%"')
    rss_count = cursor.fetchone()[0]

    print(f'Total Articles: {total}')
    print(f'  Websearch (Bing): {websearch_count} ({websearch_count/total*100:.1f}%)')
    print(f'  RSS Feeds: {rss_count} ({rss_count/total*100:.1f}%)')
    print()

    # AI Relevance Comparison
    print('-'*70)
    print('AI RELEVANCE COMPARISON')
    print('-'*70)

    cursor = conn.execute('''
        SELECT
            CASE WHEN url LIKE '%bing.com%' THEN 'Websearch' ELSE 'RSS' END as source_type,
            ai_relevant,
            COUNT(*) as count
        FROM articles
        GROUP BY source_type, ai_relevant
        ORDER BY source_type, ai_relevant
    ''')

    websearch_relevant = 0
    websearch_total = 0
    rss_relevant = 0
    rss_total = 0

    for row in cursor.fetchall():
        if row['source_type'] == 'Websearch':
            if row['ai_relevant']:
                websearch_relevant = row['count']
            websearch_total += row['count']
        else:
            if row['ai_relevant']:
                rss_relevant = row['count']
            rss_total += row['count']

    websearch_rate = (websearch_relevant / websearch_total * 100) if websearch_total > 0 else 0
    rss_rate = (rss_relevant / rss_total * 100) if rss_total > 0 else 0

    print(f'Websearch AI Relevance: {websearch_relevant}/{websearch_total} ({websearch_rate:.1f}%)')
    print(f'RSS AI Relevance: {rss_relevant}/{rss_total} ({rss_rate:.1f}%)')
    print()

    # Content Quality (article length)
    print('-'*70)
    print('CONTENT QUALITY (Article Length)')
    print('-'*70)

    cursor = conn.execute('''
        SELECT
            CASE WHEN url LIKE '%bing.com%' THEN 'Websearch' ELSE 'RSS' END as source_type,
            AVG(LENGTH(content)) as avg_content_length,
            AVG(LENGTH(summary)) as avg_summary_length,
            COUNT(*) as count
        FROM articles
        WHERE content IS NOT NULL AND content != ''
        GROUP BY source_type
    ''')

    for row in cursor.fetchall():
        print(f"{row['source_type']}:")
        print(f"  Avg content length: {row['avg_content_length']:.0f} chars")
        print(f"  Avg summary length: {row['avg_summary_length']:.0f} chars")
        print(f"  Articles with content: {row['count']}")
        print()

    # Source Diversity
    print('-'*70)
    print('SOURCE DIVERSITY')
    print('-'*70)

    cursor = conn.execute('''
        SELECT COUNT(DISTINCT source_name) as unique_sources
        FROM articles
        WHERE url NOT LIKE '%bing.com%'
    ''')
    rss_sources = cursor.fetchone()[0]

    print(f'Websearch sources: 1 (Bing News Search)')
    print(f'RSS sources: {rss_sources} unique feeds')

    cursor = conn.execute('''
        SELECT source_name, COUNT(*) as count
        FROM articles
        WHERE url NOT LIKE '%bing.com%'
        GROUP BY source_name
        ORDER BY count DESC
        LIMIT 10
    ''')

    print(f'\nTop 10 RSS sources:')
    for row in cursor.fetchall():
        print(f"  {row['source_name']}: {row['count']} articles")
    print()

    # Language Detection (basic)
    print('-'*70)
    print('LANGUAGE DISTRIBUTION (Basic Detection)')
    print('-'*70)

    cursor = conn.execute('''
        SELECT
            CASE WHEN url NOT LIKE '%bing.com%' THEN 'RSS' ELSE 'Websearch' END as source_type,
            CASE
                WHEN title LIKE '%[ Chine]%]%' OR title LIKE '% Chine%' THEN 'Chinese'
                WHEN title GLOB '*[一-龥]*' THEN 'Chinese'
                ELSE 'English'
            END as language,
            COUNT(*) as count
        FROM articles
        GROUP BY source_type, language
        ORDER BY source_type, count DESC
    ''')

    websearch_chinese = 0
    websearch_english = 0
    rss_chinese = 0
    rss_english = 0

    for row in cursor.fetchall():
        if row['source_type'] == 'Websearch':
            if row['language'] == 'Chinese':
                websearch_chinese = row['count']
            else:
                websearch_english = row['count']
        else:
            if row['language'] == 'Chinese':
                rss_chinese = row['count']
            else:
                rss_english = row['count']

    print(f'Websearch:')
    print(f'  English: {websearch_english} articles')
    print(f'  Chinese/Other: {websearch_chinese} articles')
    print()
    print(f'RSS:')
    print(f'  English: {rss_english} articles')
    print(f'  Chinese/Other: {rss_chinese} articles')
    print()

    # Recent Articles (last 24 hours)
    print('-'*70)
    print('RECENT ARTICLES (Last 24 Hours)')
    print('-'*70)

    cursor = conn.execute('''
        SELECT
            CASE WHEN url LIKE '%bing.com%' THEN 'Websearch' ELSE 'RSS' END as source_type,
            COUNT(*) as count
        FROM articles
        WHERE published_at >= datetime('now', '-1 day')
        GROUP BY source_type
    ''')

    print("Articles collected in last 24 hours:")
    for row in cursor.fetchall():
        print(f"  {row['source_type']}: {row['count']} articles")
    print()

    # Sample Articles
    print('-'*70)
    print('SAMPLE WEBSEARCH ARTICLES (Last 3)')
    print('-'*70)

    cursor = conn.execute('''
        SELECT title, source_name, published_at, summary
        FROM articles
        WHERE url LIKE '%bing.com%'
        ORDER BY published_at DESC
        LIMIT 3
    ''')

    for i, row in enumerate(cursor.fetchall(), 1):
        print(f"\n{i}. {row['title']}")
        print(f"   Source: {row['source_name']}")
        print(f"   Date: {row['published_at']}")
        if row['summary']:
            summary = row['summary'][:150] + '...' if len(row['summary']) > 150 else row['summary']
            print(f"   Summary: {summary}")

    print()
    print('-'*70)
    print('SAMPLE RSS ARTICLES (Last 3)')
    print('-'*70)

    cursor = conn.execute('''
        SELECT title, source_name, published_at, summary
        FROM articles
        WHERE url NOT LIKE '%bing.com%'
        ORDER BY published_at DESC
        LIMIT 3
    ''')

    for i, row in enumerate(cursor.fetchall(), 1):
        print(f"\n{i}. {row['title']}")
        print(f"   Source: {row['source_name']}")
        print(f"   Date: {row['published_at']}")
        if row['summary']:
            summary = row['summary'][:150] + '...' if len(row['summary']) > 150 else row['summary']
            print(f"   Summary: {summary}")

    # Recommendations
    print()
    print('='*70)
    print('RECOMMENDATIONS')
    print('='*70)

    if websearch_rate > rss_rate:
        print(f'✅ Websearch has better AI relevance ({websearch_rate:.1f}% vs {rss_rate:.1f}%)')
    else:
        print(f'⚠️  RSS has better AI relevance ({rss_rate:.1f}% vs {websearch_rate:.1f}%)')

    if rss_chinese > rss_english * 0.5:
        print(f'⚠️  RSS has many Chinese articles ({rss_chinese} vs {rss_english} English)')
        print('   Consider filtering to English-only sources')

    if websearch_total < 100:
        print(f'⚠️  Websearch collection is low ({websearch_total} articles)')
        print('   Consider running: ai-news collect --websearch --topics AI,LLM')

    print()
    print('='*70)


if __name__ == '__main__':
    evaluate_quality()
