#!/usr/bin/env python
"""
Evaluate and rank article quality from combined websearch+RSS collection.

Scoring criteria:
1. AI relevance (0-30 points)
2. Content length (0-20 points)
3. Source quality (0-15 points)
4. Topic match (0-20 points)
5. Freshness (0-15 points)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ai_news.database import Database
import sqlite3
from datetime import datetime, timedelta


def calculate_quality_score(article, topics=None) -> dict:
    """Calculate quality score for an article.

    Returns:
        Dict with score and breakdown
    """
    score = 0
    breakdown = {}

    # 1. AI Relevance (0-30 points)
    if article.ai_relevant:
        relevance_score = 30
        breakdown['ai_relevance'] = 30
    else:
        relevance_score = 0
        breakdown['ai_relevance'] = 0
    score += relevance_score

    # 2. Content Length (0-20 points)
    content_length = len(article.content) if article.content else 0
    if content_length == 0:
        length_score = 0
    elif content_length < 200:
        length_score = 5  # Very short
    elif content_length < 500:
        length_score = 10  # Short
    elif content_length < 1000:
        length_score = 15  # Medium
    else:
        length_score = 20  # Long, detailed
    breakdown['content_length'] = length_score
    score += length_score

    # 3. Source Quality (0-15 points)
    source = article.source_name.lower()
    if any(x in source for x in ['techcrunch', 'arxiv', 'mit', 'wired', 'verge']):
        source_score = 15  # Premium tech sources
    elif any(x in source for x in ['reuters', 'bbc', 'guardian', 'bloomberg']):
        source_score = 12  # Major news outlets
    elif 'bing' in source:
        source_score = 10  # Websearch (curated)
    elif any(x in source for x in ['hacker', 'reddit']):
        source_score = 8  # Community sources
    elif any(x in source for x in ['36kr', 'china', 'ifanr']):
        source_score = 5  # Non-English sources
    else:
        source_score = 7  # Other sources
    breakdown['source_quality'] = source_score
    score += source_score

    # 4. Topic Match (0-20 points)
    if topics:
        title_lower = article.title.lower()
        summary_lower = (article.summary or '').lower()

        matches = sum(1 for topic in topics if topic.lower() in title_lower)
        matches += sum(1 for topic in topics if topic.lower() in summary_lower)

        if matches >= 3:
            topic_score = 20
        elif matches == 2:
            topic_score = 15
        elif matches == 1:
            topic_score = 10
        else:
            topic_score = 5
    else:
        topic_score = 10  # Neutral score if no topics specified
    breakdown['topic_match'] = topic_score
    score += topic_score

    # 5. Freshness (0-15 points)
    if article.published_at:
        # Handle timezone-aware datetimes
        if article.published_at.tzinfo:
            published_at = article.published_at.astimezone(None).replace(tzinfo=None)
        else:
            published_at = article.published_at
        article_age = (datetime.now() - published_at).days
        if article_age <= 1:
            freshness_score = 15  # Very fresh
        elif article_age <= 3:
            freshness_score = 12  # Fresh
        elif article_age <= 7:
            freshness_score = 9  # This week
        elif article_age <= 30:
            freshness_score = 6  # This month
        else:
            freshness_score = 3  # Older
    else:
        freshness_score = 0
    breakdown['freshness'] = freshness_score
    score += freshness_score

    return {
        'total_score': score,
        'max_score': 100,
        'breakdown': breakdown,
        'grade': get_grade(score)
    }


def get_grade(score: int) -> str:
    """Convert score to letter grade."""
    if score >= 90:
        return 'A+'
    elif score >= 80:
        return 'A'
    elif score >= 70:
        return 'B'
    elif score >= 60:
        return 'C'
    else:
        return 'D'


def evaluate_collection(topics=None, limit=100):
    """Evaluate and rank recent articles.

    Args:
        topics: List of topics to filter by (optional)
        limit: Number of articles to evaluate
    """
    db = Database('data/production/ai_news.db')
    conn = sqlite3.connect('data/production/ai_news.db')
    conn.row_factory = sqlite3.Row

    print('='*80)
    print('ARTICLE QUALITY EVALUATION')
    print('='*80)
    print()

    # Get recent articles
    if topics:
        print(f"Topics: {', '.join(topics)}")
        print(f"Limit: {limit} articles")
        print()

        # Filter by topics
        topics_lower = [t.lower() for t in topics]
        cursor = conn.execute('''
            SELECT * FROM articles
            WHERE LOWER(title) LIKE ? OR LOWER(summary) LIKE ?
            ORDER BY published_at DESC
            LIMIT ?
        ''', (f'%{topics_lower[0]}%', f'%{topics_lower[0]}%', limit))
    else:
        cursor = conn.execute('''
            SELECT * FROM articles
            ORDER BY published_at DESC
            LIMIT ?
        ''', (limit,))

    articles = []
    for row in cursor.fetchall():
        from src.ai_news.database import Article
        article = Article(
            id=row['id'],
            title=row['title'],
            content=row['content'],
            summary=row['summary'],
            url=row['url'],
            author=row['author'] or '',
            published_at=datetime.fromisoformat(row['published_at']) if row['published_at'] else None,
            source_name=row['source_name'] or '',
            category=row['category'] or '',
            region=row['region'] or 'global',
            ai_relevant=bool(row['ai_relevant']),
            ai_keywords_found=row['ai_keywords_found'].split(',') if row['ai_keywords_found'] else []
        )
        articles.append(article)

    print(f"Evaluating {len(articles)} articles...\n")

    # Score all articles
    scored_articles = []
    for article in articles:
        score_data = calculate_quality_score(article, topics)
        scored_articles.append({
            'article': article,
            'score': score_data
        })

    # Sort by score (descending)
    scored_articles.sort(key=lambda x: x['score']['total_score'], reverse=True)

    # Display rankings
    print('='*80)
    print('TOP 20 HIGHEST QUALITY ARTICLES')
    print('='*80)
    print()

    for i, item in enumerate(scored_articles[:20], 1):
        article = item['article']
        score_data = item['score']

        grade = score_data['grade']
        total = score_data['total_score']
        breakdown = score_data['breakdown']

        # Determine source type
        source_type = "🔍 Websearch" if 'bing' in article.url.lower() else "📡 RSS"

        print(f"{i}. [{grade}] {total}/100 pts {source_type}")
        print(f"   {article.title[:80]}")
        print(f"   Source: {article.source_name} | Date: {article.published_at.strftime('%Y-%m-%d') if article.published_at else 'Unknown'}")
        print(f"   Scores: AI+{breakdown['ai_relevance']} Len+{breakdown['content_length']} Src+{breakdown['source_quality']} Topic+{breakdown['topic_match']} Fresh+{breakdown['freshness']}")
        print()

    # Statistics
    print('='*80)
    print('QUALITY STATISTICS')
    print('='*80)
    print()

    scores = [item['score']['total_score'] for item in scored_articles]
    grades = [item['score']['grade'] for item in scored_articles]

    avg_score = sum(scores) / len(scores) if scores else 0

    print(f"Average Score: {avg_score:.1f}/100")
    print(f"Highest Score: {max(scores) if scores else 0}")
    print(f"Lowest Score: {min(scores) if scores else 0}")
    print()

    # Grade distribution
    grade_counts = {}
    for grade in grades:
        grade_counts[grade] = grade_counts.get(grade, 0) + 1

    print("Grade Distribution:")
    for grade in ['A+', 'A', 'B', 'C', 'D']:
        count = grade_counts.get(grade, 0)
        pct = (count / len(grades) * 100) if grades else 0
        bar = '█' * int(pct / 5)
        print(f"  {grade:2}: {count:3} articles ({pct:5.1f}%) {bar}")

    print()

    # Source breakdown
    websearch_articles = [item for item in scored_articles if 'bing' in item['article'].url.lower()]
    rss_articles = [item for item in scored_articles if 'bing' not in item['article'].url.lower()]

    websearch_avg = sum([item['score']['total_score'] for item in websearch_articles]) / len(websearch_articles) if websearch_articles else 0
    rss_avg = sum([item['score']['total_score'] for item in rss_articles]) / len(rss_articles) if rss_articles else 0

    print(f"Websearch Articles: {len(websearch_articles)} | Avg Score: {websearch_avg:.1f}/100")
    print(f"RSS Articles: {len(rss_articles)} | Avg Score: {rss_avg:.1f}/100")
    print()

    # Recommendations
    print('='*80)
    print('RECOMMENDATIONS')
    print('='*80)
    print()

    if websearch_avg > rss_avg:
        diff = websearch_avg - rss_avg
        print(f"✅ Websearch has higher quality (+{diff:.1f} points average)")
        print(f"   Prefer websearch for highest relevance and topic matching")
    elif rss_avg > websearch_avg:
        diff = rss_avg - websearch_avg
        print(f"✅ RSS has higher quality (+{diff:.1f} points average)")
        print(f"   Prefer RSS for deeper content and more detailed coverage")

    print()
    print(f"For best quality digests, use:")
    print(f"  ai-news digest --type topic --topics {','.join(topics) if topics else 'AI'} --min-quality 70")
    print()
    print('='*80)


if __name__ == '__main__':
    topics = ['LLM', 'GPT']  # Specify topics to evaluate
    evaluate_collection(topics=topics, limit=200)
