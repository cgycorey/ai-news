"""Markdown generator for AI news articles and digests."""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import re
from pathlib import Path

from .database import Article, Database


class MarkdownGenerator:
    """Generate markdown files from collected news articles."""
    
    def __init__(self, database: Database):
        self.database = database
    
    def clean_content_for_markdown(self, content: str) -> str:
        """Clean content for safe markdown rendering."""
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        
        # Escape problematic markdown characters
        content = content.replace('#', '\\#')
        
        return content.strip()
    
    def generate_article_summary(self, article: Article, max_length: int = 300) -> str:
        """Generate a better summary for an article."""
        if article.summary and len(article.summary) > 50:
            return article.summary
        
        # Try to extract first meaningful sentences
        content = article.content or article.title
        
        # Split into sentences and take first few
        sentences = re.split(r'[.!?]+', content)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        if meaningful_sentences:
            summary = '. '.join(meaningful_sentences[:2])
            if len(summary) > max_length:
                summary = summary[:max_length].rsplit(' ', 1)[0] + '...'
            return summary + '.'
        
        return content[:max_length] + ('...' if len(content) > max_length else '')
    
    def format_article_md(self, article: Article) -> str:
        """Format a single article in markdown."""
        # Generate a good summary
        summary = self.generate_article_summary(article)
        
        # Clean content for markdown
        clean_content = self.clean_content_for_markdown(article.content)
        
        # Format AI keywords
        keywords_str = ""
        if article.ai_keywords_found:
            keywords_str = f"**AI Keywords:** {', '.join(article.ai_keywords_found)}\n\n"
        
        # Article metadata
        metadata = f"""**Source:** [{article.source_name}]({article.url})  
**Author:** {article.author or 'Unknown'}  
**Published:** {article.published_at.strftime('%Y-%m-%d %H:%M') if article.published_at else 'Unknown'}  
**Category:** {article.category}  
**AI Relevant:** {'✅ Yes' if article.ai_relevant else '❌ No'}

{keywords_str}"""
        
        # Build full article markdown
        article_md = f"""### {article.title}

{metadata}

**Summary:** {summary}

---
"""
        
        return article_md
    
    def generate_daily_digest(self, date: datetime, ai_only: bool = False) -> str:
        """Generate a daily digest markdown."""
        # Get articles for the date
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)
        
        articles = self.database.get_articles(limit=100)
        date_articles = []
        for a in articles:
            if a.published_at:
                # Make both dates comparable by handling timezone info
                if a.published_at.tzinfo:
                    article_date = a.published_at.astimezone(None).replace(tzinfo=None)
                else:
                    article_date = a.published_at
                
                if start_date <= article_date < end_date:
                    date_articles.append(a)
        
        if ai_only:
            date_articles = [a for a in date_articles if a.ai_relevant]
        
        if not date_articles:
            return f"# AI News Digest - {date.strftime('%Y-%m-%d')}\n\n*No articles found for this date.*"
        
        # Separate AI and non-AI articles
        ai_articles = [a for a in date_articles if a.ai_relevant]
        other_articles = [a for a in date_articles if not a.ai_relevant]
        
        digest = f"""# AI News Digest - {date.strftime('%Y-%m-%d')}

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*

"""
        
        # AI-Related Articles Section
        if ai_articles:
            digest += f"""## 🤖 AI-Related Articles ({len(ai_articles)})

"""
            for i, article in enumerate(ai_articles, 1):
                digest += f"""### {i}. {article.title}

**Source:** {article.source_name} | **Category:** {article.category} | **Time:** {article.published_at.strftime('%H:%M') if article.published_at else 'Unknown'}

{self.generate_article_summary(article)}

**Read more:** [{article.url}]({article.url})

"""
                if article.ai_keywords_found:
                    digest += f"**AI Keywords:** {', '.join(article.ai_keywords_found)}\n\n"
        
        # Other Articles Section (if not ai_only)
        if not ai_only and other_articles:
            digest += f"""## 📰 Other Tech News ({len(other_articles)})

"""
            for i, article in enumerate(other_articles, 1):
                digest += f"""### {i}. {article.title}

**Source:** {article.source_name} | **Category:** {article.category} | **Time:** {article.published_at.strftime('%H:%M') if article.published_at else 'Unknown'}

{self.generate_article_summary(article)}

**Read more:** [{article.url}]({article.url})

"""
        
        # Statistics
        total_sources = len(set(a.source_name for a in date_articles))
        ai_percentage = (len(ai_articles) / len(date_articles) * 100) if date_articles else 0
        
        digest += f"""---

## 📊 Statistics

- **Total Articles:** {len(date_articles)}
- **AI-Related:** {len(ai_articles)} ({ai_percentage:.1f}%)
- **Sources:** {total_sources}
- **Date Range:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}

## 🔍 Top AI Keywords

"""
        
        # Count and display top AI keywords
        all_keywords = []
        for article in ai_articles:
            all_keywords.extend(article.ai_keywords_found or [])
        
        if all_keywords:
            keyword_counts = {}
            for keyword in all_keywords:
                keyword_counts[keyword.lower()] = keyword_counts.get(keyword.lower(), 0) + 1
            
            top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for keyword, count in top_keywords:
                digest += f"- **{keyword.title()}**: {count} mentions\n"
        else:
            digest += "*No AI keywords found.*\n"
        
        digest += "\n---\n*Generated by AI News Collector*"
        
        return digest
    
    def generate_weekly_digest(self, start_date: datetime) -> str:
        """Generate a weekly digest markdown."""
        end_date = start_date + timedelta(days=7)
        
        articles = self.database.get_articles(limit=500)
        week_articles = []
        for a in articles:
            if a.published_at:
                if a.published_at.tzinfo:
                    article_date = a.published_at.astimezone(None).replace(tzinfo=None)
                else:
                    article_date = a.published_at
                
                if start_date <= article_date < end_date:
                    week_articles.append(a)
        
        if not week_articles:
            return f"# Weekly AI News Digest - {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n*No articles found for this week.*"
        
        # Group by category
        categories = {}
        ai_articles = []
        
        for article in week_articles:
            cat = article.category or 'general'
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(article)
            
            if article.ai_relevant:
                ai_articles.append(article)
        
        digest = f"""# Weekly AI News Digest
{start_date.strftime('%B %d, %Y')} - {end_date.strftime('%B %d, %Y')}

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*

"""
        
        # Key Highlights Section
        if ai_articles:
            digest += f"""## 🎯 Key Highlights

"""
            # Get top articles by AI relevance (most keywords)
            scored_articles = [(a, len(a.ai_keywords_found or [])) for a in ai_articles]
            scored_articles.sort(key=lambda x: x[1], reverse=True)
            
            for i, (article, score) in enumerate(scored_articles[:5], 1):
                digest += f"""### {i}. {article.title}

**Source:** {article.source_name} | **Relevance Score:** {score} AI keywords

{self.generate_article_summary(article)}

**Read more:** [{article.url}]({article.url})

"""
        
        # Category breakdown
        digest += f"""## 📂 By Category

"""
        for category, cat_articles in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
            ai_count = len([a for a in cat_articles if a.ai_relevant])
            digest += f"""### {category.title()} ({len(cat_articles)} articles, {ai_count} AI-related)

"""
            for article in cat_articles[:3]:  # Top 3 per category
                ai_indicator = "🤖 " if article.ai_relevant else ""
                digest += f"""{ai_indicator}**{article.title}**
*{article.source_name}* - {self.generate_article_summary(article, max_length=150)}

**Read more:** [{article.url}]({article.url})

"""
        
        # Statistics
        total_sources = len(set(a.source_name for a in week_articles))
        ai_percentage = (len(ai_articles) / len(week_articles) * 100) if week_articles else 0
        
        digest += f"""## 📊 Weekly Statistics

- **Total Articles:** {len(week_articles)}
- **AI-Related:** {len(ai_articles)} ({ai_percentage:.1f}%)
- **Sources:** {total_sources}
- **Categories:** {len(categories)}

"""
        
        return digest
    
    def generate_topic_analysis(self, topic: str, days: int = 7) -> str:
        """Generate analysis for a specific topic."""
        start_date = datetime.now() - timedelta(days=days)
        
        # Search for articles related to topic
        articles = self.database.search_articles(topic, limit=50)
        
        # Filter by date
        recent_articles = []
        for a in articles:
            if not a.published_at:
                recent_articles.append(a)  # Include articles without dates
            else:
                if a.published_at.tzinfo:
                    article_date = a.published_at.astimezone(None).replace(tzinfo=None)
                else:
                    article_date = a.published_at
                
                if article_date >= start_date:
                    recent_articles.append(a)
        
        if not recent_articles:
            return f"# Topic Analysis: {topic}\n\n*No articles found for '{topic}' in the last {days} days.*"
        
        digest = f"""# Topic Analysis: {topic}
*Last {days} days* - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}

"""
        
        # Articles by sentiment/time
        ai_articles = [a for a in recent_articles if a.ai_relevant]
        
        digest += f"""## 📈 Overview

- **Total Articles:** {len(recent_articles)}
- **AI-Related:** {len(ai_articles)}
- **Coverage Period:** {start_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}

"""
        
        digest += f"""## 📰 Articles ({len(recent_articles)})

"""
        for i, article in enumerate(recent_articles, 1):
            ai_indicator = "🤖 " if article.ai_relevant else ""
            date_str = article.published_at.strftime('%Y-%m-%d') if article.published_at else 'Unknown'
            
            digest += f"""### {i}. {ai_indicator}{article.title}

**Source:** {article.source_name} | **Date:** {date_str} | **Category:** {article.category}

{self.generate_article_summary(article)}

**Read more:** [{article.url}]({article.url})

"""
            if article.ai_keywords_found:
                digest += f"**AI Keywords:** {', '.join(article.ai_keywords_found)}\n\n"
        
        return digest
    
    def save_digest_to_file(self, content: str, filename: str, output_dir: str = "digests") -> Path:
        """Save digest content to a markdown file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        file_path = output_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return file_path