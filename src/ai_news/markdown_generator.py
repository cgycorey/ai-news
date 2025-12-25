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
    
    def generate_spacy_topic_digest(
        self, 
        topics: List[str], 
        scored_articles: List, 
        days: int
    ) -> str:
        """
        Generate digest with spaCy-powered relevance grouping.
        
        Groups scored articles by confidence levels:
        - Strong Match: ≥0.85
        - Moderate Match: ≥0.70 to <0.85
        - Related: ≥0.50 to <0.70
        
        Within each group, articles are sorted chronologically (newest first).
        
        Args:
            topics: List of topic keywords
            scored_articles: List of ScoredArticle objects with confidence and matched_entities
            days: Number of days for digest
            
        Returns:
            Complete markdown digest as string
        """
        import time
        start_time = time.time()
        
        # Group articles by confidence levels
        strong_match = []
        moderate_match = []
        related = []
        
        for scored_article in scored_articles:
            if scored_article.confidence >= 0.85:
                strong_match.append(scored_article)
            elif scored_article.confidence >= 0.70:
                moderate_match.append(scored_article)
            elif scored_article.confidence >= 0.50:
                related.append(scored_article)
        
        # Sort each group chronologically (newest first by published_at)
        def sort_by_published_date(article):
            published_at = article.article.get("published_at")
            if published_at:
                # Handle timezone-aware datetimes
                if hasattr(published_at, 'tzinfo') and published_at.tzinfo:
                    return published_at.astimezone(None).replace(tzinfo=None)
                return published_at
            return datetime.min
        
        strong_match.sort(key=sort_by_published_date, reverse=True)
        moderate_match.sort(key=sort_by_published_date, reverse=True)
        related.sort(key=sort_by_published_date, reverse=True)
        
        # Generate header
        topics_str = ", ".join(topics)
        digest = f"""# AI News Digest: {topics_str} (Last {days} Days)

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*  
**Topics:** {topics_str}  
**Method:** spaCy semantic analysis

"""
        
        # Add Strong Match section
        if strong_match:
            digest += f"""## 🔥 Strong Match ({len(strong_match)} articles)
Confidence: ≥85% - Both topics clearly present with semantic similarity

"""
            for i, scored_article in enumerate(strong_match, 1):
                digest += self._format_scored_article(scored_article, i)
        
        # Add Moderate Match section
        if moderate_match:
            digest += f"""## 📊 Moderate Match ({len(moderate_match)} articles)
Confidence: 70-84% - Topics discussed with good confidence

"""
            for i, scored_article in enumerate(moderate_match, 1):
                digest += self._format_scored_article(scored_article, i)
        
        # Add Related section
        if related:
            digest += f"""## 💡 Related ({len(related)} articles)
Confidence: 50-69% - Topics mentioned but not central

"""
            for i, scored_article in enumerate(related, 1):
                digest += self._format_scored_article(scored_article, i)
        
        # If no articles found
        if not strong_match and not moderate_match and not related:
            digest += "*No articles found matching the specified topics and confidence thresholds.*\n\n"
        
        # Calculate processing time
        processing_time = time.time() - start_time
        total_articles = len(scored_articles)
        
        # Add statistics section
        digest += f"""---

## 📊 Statistics

- **Total Articles:** {total_articles} articles
- **Processing Time:** {processing_time:.1f}s
- **Cache:** miss (next run: ~0.1s)

---
*Generated by AI News Collector with spaCy-powered relevance grouping*
"""
        return digest
    
    def _format_scored_article(self, scored_article, index: int) -> str:
        """
        Format a scored article for markdown output.
        
        Args:
            scored_article: ScoredArticle object
            index: Article number in the list
            
        Returns:
            Formatted markdown string
        """
        article = scored_article.article
        confidence_pct = scored_article.confidence * 100
        
        # Extract metadata
        title = article.get("title", "Untitled")
        url = article.get("url", "")
        source_name = article.get("source_name", "Unknown")
        author = article.get("author", "Unknown")
        published_at = article.get("published_at")
        summary = article.get("summary", article.get("content", ""))[:300]
        
        # Format published date
        if published_at:
            if isinstance(published_at, datetime):
                published_str = published_at.strftime('%Y-%m-%d %H:%M')
            else:
                published_str = str(published_at)
        else:
            published_str = "Unknown"
        
        # Format matched entities
        matched_entities = sorted(scored_article.matched_entities) if scored_article.matched_entities else []
        entities_str = ", ".join(matched_entities) if matched_entities else "None"
        
        # Build article markdown
        md = f"""### {index}. {title}
**Source:** [{source_name}]({url}) | **Author:** {author} | **Published:** {published_str}
**Confidence:** {confidence_pct:.1f}% | **Matched Entities:** {entities_str}

**Summary:** {summary}

"""
        return md