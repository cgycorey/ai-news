#!/usr/bin/env python3
"""
Simple AI News Demo - Shows today's real AI news from the database.

Run with: uv run python show-today-news.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from ai_news.database import Database
from ai_news.intelligence_db import IntelligenceDB


def get_sentiment_label(score):
    """Convert sentiment score to readable label."""
    if score is None:
        return "UNKNOWN"
    elif score > 0.1:
        return "POSITIVE"
    elif score < -0.1:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


def format_confidence(score):
    """Format confidence score as percentage."""
    return f"{score:.2f}" if score is not None else "0.00"


def main():
    """Show today's AI news from real database."""
    # Get current date and time
    now = datetime.now()
    current_date_str = now.strftime("%A, %B %d, %Y")
    current_time_str = now.strftime("%I:%M %p").lstrip("0").upper()
    
    print("=" * 60)
    print(f"TODAY'S AI NEWS - {current_date_str}")
    print("=" * 60)
    print(f"Current Time: {current_time_str}")
    print(f"Database Last Updated: {now.strftime('%Y-%m-%d %H:%M')}")
    print()
    
    # Initialize database connections
    try:
        db = Database("ai_news.db")
        intell_db = IntelligenceDB("ai_news.db")
        print(f"🔥 Connected to database successfully!")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return
    
    # Get database statistics
    stats = db.get_stats()
    print(f"📊 Database Stats: {stats['total_articles']} total articles, ")
    print(f"                  {stats['ai_relevant_articles']} AI-relevant ({stats['ai_relevance_rate']})")
    print(f"                  from {stats['sources_count']} different sources")
    print(f"                  📅 Data as of {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Show top entities first
    try:
        all_entities = intell_db.get_entities(limit=20)
        if all_entities:
            print(f"🏢 Top AI Entities in System (as of {current_time_str}):")
            entity_types = {}
            for entity in all_entities:
                entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1
                if entity.confidence_score > 0.8:  # Show high-confidence entities
                    print(f"  • {entity.name} ({entity.entity_type}) - {format_confidence(entity.confidence_score * 100)}% confidence")
            print()
    except Exception as e:
        print(f"Error getting entity statistics: {e}")
    
    # Get recent articles (last 24 hours, or fallback to latest 10)
    start_time = now - timedelta(hours=24)
    recent_articles = db.get_articles_by_date_range(start_time, limit=10)
    
    # Count today's articles (handle timezone-aware datetimes)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_articles = []
    for article in recent_articles:
        if article.published_at:
            # Handle timezone-aware datetimes
            if article.published_at.tzinfo is not None:
                published_time = article.published_at.replace(tzinfo=None)
            else:
                published_time = article.published_at
            if published_time >= today_start:
                today_articles.append(article)
    
    if not recent_articles:
        # Fallback: get latest 10 AI-relevant articles
        recent_articles = db.get_articles(limit=10, ai_only=True)
        print("ℹ️  No articles from last 24h, showing latest AI-relevant articles:")
    else:
        print(f"📰 Found {len(recent_articles)} articles from last 24 hours")
        if today_articles:
            print(f"🔥 {len(today_articles)} articles published TODAY ({current_date_str})")
        else:
            print(f"ℹ️  No articles published yet today ({current_date_str})")
    
    print()
    
    # Display articles with their entities and sentiment
    for i, article in enumerate(recent_articles[:8], 1):
        print(f"{i}. ARTICLE: {article.title}")
        
        # Format date nicely with relative time and today indicator
        if article.published_at:
            # Handle timezone-aware datetimes
            if article.published_at.tzinfo is not None:
                published_time = article.published_at.replace(tzinfo=None)
            else:
                published_time = article.published_at
            
            date_str = published_time.strftime("%Y-%m-%d %H:%M")
            
            # Calculate relative time
            time_diff = now - published_time
            if time_diff.total_seconds() < 60:  # Less than 1 minute
                relative_time = "just now"
            elif time_diff.total_seconds() < 3600:  # Less than 1 hour
                mins = int(time_diff.total_seconds() / 60)
                relative_time = f"{mins} minute{'s' if mins != 1 else ''} ago"
            elif time_diff.total_seconds() < 86400:  # Less than 1 day
                hours = int(time_diff.total_seconds() / 3600)
                relative_time = f"{hours} hour{'s' if hours != 1 else ''} ago"
            else:
                relative_time = f"{time_diff.days} day{'s' if time_diff.days != 1 else ''} ago"
            
            # Highlight today's articles
            if published_time.date() == now.date():
                date_str = f"🔥 TODAY | {date_str} ({relative_time})"
            else:
                date_str = f"{date_str} ({relative_time})"
        else:
            date_str = "Unknown date"
        
        print(f"   📅 {article.source_name or 'Unknown'} | {date_str}")
        
        # Show AI relevance
        ai_status = "YES" if article.ai_relevant else "NO"
        print(f"   AI Relevance: {ai_status}")
        
        # Show AI keywords found
        if article.ai_keywords_found:
            keywords = ", ".join(article.ai_keywords_found[:5])
            if len(article.ai_keywords_found) > 5:
                keywords += f" ... (+{len(article.ai_keywords_found) - 5} more)"
            print(f"   AI Keywords: {keywords}")
        
        # Try to get entities for this article
        try:
            entities_with_scores = intell_db.get_article_entities(article.id)
            if entities_with_scores:
                entities_str = ", ".join([
                    f"{entity.name} ({entity.entity_type})" 
                    for entity, score in entities_with_scores[:2]
                ])
                if len(entities_with_scores) > 2:
                    entities_str += f" ... (+{len(entities_with_scores) - 2} more)"
                print(f"   Entities: {entities_str}")
            else:
                # Try to find entities mentioned in the title/text
                article_lower = article.title.lower()
                mentioned_entities = []
                for entity in all_entities[:10]:  # Check against top entities
                    if entity.name.lower() in article_lower:
                        mentioned_entities.append(f"{entity.name} ({entity.entity_type})")
                        break  # Just show first match to avoid too much noise
                if mentioned_entities:
                    print(f"   Key Entity: {mentioned_entities[0]} (detected in title)")
                else:
                    print(f"   Entities: Entity analysis pending")
        except Exception as e:
            print(f"   Entities: Analysis in progress")
        
        # Show article summary if available
        if article.summary and len(article.summary.strip()) > 20:
            summary = article.summary[:150] + "..." if len(article.summary) > 150 else article.summary
            print(f"   📝 Summary: {summary}")
        
        # Show URL if available
        if article.url:
            print(f"   🔗 Read more: {article.url}")
        
        print()
    
    # Show additional statistics
    try:
        all_entities = intell_db.get_entities(limit=100)
        if all_entities:
            entity_types = {}
            for entity in all_entities:
                entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1
            
            print(f"Complete Entity Statistics:")
            for entity_type, count in sorted(entity_types.items()):
                print(f"   {entity_type.capitalize()}: {count} entities")
            print()
    except Exception as e:
        print(f"Error getting entity statistics: {e}")
    
    # Show some interesting insights with current timestamp
    print("🚀 AI News Insights (as of today):")
    print(f"• The system tracks {stats['sources_count']} different news sources")
    print(f"• {stats['ai_relevance_rate']} of articles are AI-relevant")
    print(f"• Articles are automatically categorized by AI keyword detection")
    print(f"• Entity extraction identifies companies, products, technologies, and people")
    print(f"• Database contains {stats['total_articles']} articles with real-time analysis")
    print(f"• Last system refresh: {current_time_str}")
    print(f"• Today's date: {current_date_str}")
    print()
    
    print("=" * 60)
    print(f"🔥 AI News System - Real Fresh News for {current_date_str}! 🔥")
    print("=" * 60)
    print()
    print("This shows REAL AI news collected from actual sources")
    print("📰 Articles are collected from 16+ different tech news sources")
    print("🤖 Entities are extracted using advanced NLP techniques")
    print("🎯 AI relevance is determined by keyword matching and ML")
    print(f"⏰ Database updates every 6 hours (last check: {current_time_str})")
    print(f"🗓️  All timestamps above show freshness relative to now ({current_time_str})")
    print()
    print("Want to see more? Try:")
    print(f"   python3 -c \"from ai_news.database import Database; db=Database('ai_news.db'); print([a.title for a in db.get_articles(limit=10)])\"")
    print()


if __name__ == "__main__":
    main()