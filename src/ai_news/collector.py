"""Simple news collection module for RSS feeds using only standard library."""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List
import time
import re
import html

from .config import FeedConfig
from .database import Article, Database


class SimpleCollector:
    """Collects news from RSS feeds using only standard library."""
    
    def __init__(self, database: Database):
        self.database = database
        self.headers = {
            'User-Agent': 'AI-News-Collector/1.0 (Simple RSS Reader)'
        }
    
    def clean_html(self, html_content: str) -> str:
        """Remove HTML tags using simple regex."""
        if not html_content:
            return ""
        
        # Unescape HTML entities
        content = html.unescape(html_content)
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', ' ', content)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    def is_ai_relevant(self, title: str, content: str, keywords: List[str]) -> tuple[bool, List[str]]:
        """Check if content is AI-related based on keywords."""
        text = (title + " " + content).lower()
        found_keywords = []
        
        for keyword in keywords:
            if keyword.lower() in text:
                found_keywords.append(keyword)
        
        return len(found_keywords) > 0, found_keywords
    
    def create_summary(self, content: str, max_length: int = 200) -> str:
        """Create a simple summary by truncating content."""
        if not content:
            return ""
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        if len(content) <= max_length:
            return content
        
        # Try to end at a sentence boundary
        truncated = content[:max_length]
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        
        last_boundary = max(last_period, last_exclamation, last_question)
        
        if last_boundary > max_length * 0.7:  # Only cut if we have at least 70% of content
            return truncated[:last_boundary + 1]
        
        return truncated + "..."
    
    def fetch_rss_feed(self, url: str) -> ET.Element | None:
        """Fetch and parse RSS feed."""
        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
            
            # Parse XML
            root = ET.fromstring(content)
            return root
            
        except Exception as e:
            print(f"Error fetching RSS feed from {url}: {e}")
            return None
    
    def parse_rss_item(self, item: ET.Element) -> dict:
        """Parse a single RSS item."""
        data = {}
        
        # Handle different RSS formats
        title = item.find('title')
        if title is not None:
            data['title'] = title.text or ""
        
        link = item.find('link')
        if link is not None:
            data['link'] = link.text or ""
        
        # Try different content fields
        content = item.find('content:encoded', {'content': 'http://purl.org/rss/1.0/modules/content/'})
        if content is None:
            content = item.find('description')
        if content is not None:
            data['content'] = content.text or ""
        
        # Author
        author = item.find('author')
        if author is not None:
            data['author'] = author.text or ""
        
        # Date (try different fields)
        date_fields = ['pubDate', 'published', 'dc:date']
        for field in date_fields:
            date_elem = item.find(field)
            if date_elem is not None and date_elem.text:
                data['date'] = date_elem.text
                break
        
        return data
    
    def parse_date(self, date_str: str) -> datetime | None:
        """Parse date string into datetime object."""
        if not date_str:
            return None
        
        # Common RSS date formats
        formats = [
            '%a, %d %b %Y %H:%M:%S %Z',  # RFC 822
            '%a, %d %b %Y %H:%M:%S %z',  # RFC 822 with timezone
            '%Y-%m-%dT%H:%M:%S%z',       # ISO 8601
            '%Y-%m-%dT%H:%M:%SZ',        # ISO 8601 UTC
            '%Y-%m-%d %H:%M:%S',         # Simple format
            '%Y-%m-%d',                  # Date only
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        return None
    
    def fetch_feed(self, feed_config: FeedConfig, max_articles: int = 50) -> List[Article]:
        """Fetch articles from a single RSS feed."""
        articles = []
        
        print(f"  Fetching from {feed_config.name}...")
        
        root = self.fetch_rss_feed(feed_config.url)
        if root is None:
            return articles
        
        # Find items (handle both RSS and Atom formats)
        items = []
        
        # RSS format
        channel = root.find('channel')
        if channel is not None:
            items = channel.findall('item')
        else:
            # Atom format
            items = root.findall('entry')
        
        for item in items[:max_articles]:
            try:
                if item.tag == 'entry':  # Atom format
                    data = self.parse_atom_entry(item)
                else:  # RSS format
                    data = self.parse_rss_item(item)
                
                title = data.get('title', '')
                url = data.get('link', '')
                
                if not title or not url:
                    continue
                
                # Clean HTML
                clean_content = self.clean_html(data.get('content', ''))
                
                # Get published date
                published_at = self.parse_date(data.get('date', ''))
                
                # Check AI relevance
                is_ai, keywords_found = self.is_ai_relevant(title, clean_content, feed_config.ai_keywords)
                
                # Create summary
                summary = self.create_summary(clean_content)
                
                article = Article(
                    title=title,
                    content=clean_content,
                    summary=summary,
                    url=url,
                    author=data.get('author', ''),
                    published_at=published_at,
                    source_name=feed_config.name,
                    category=feed_config.category,
                    ai_relevant=is_ai,
                    ai_keywords_found=keywords_found
                )
                
                articles.append(article)
                
            except Exception as e:
                print(f"    Error processing article: {e}")
                continue
        
        print(f"  Found {len(articles)} articles")
        return articles
    
    def parse_atom_entry(self, entry: ET.Element) -> dict:
        """Parse an Atom entry."""
        data = {}
        
        title = entry.find('{http://www.w3.org/2005/Atom}title')
        if title is not None:
            data['title'] = title.text or ""
        
        # Atom links
        links = entry.findall('{http://www.w3.org/2005/Atom}link')
        for link in links:
            if link.get('type') == 'text/html' or not link.get('type'):
                data['link'] = link.get('href', '')
                break
        
        # Content
        content = entry.find('{http://www.w3.org/2005/Atom}content')
        if content is None:
            content = entry.find('{http://www.w3.org/2005/Atom}summary')
        if content is not None:
            data['content'] = content.text or ""
        
        # Author
        author = entry.find('{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name')
        if author is not None:
            data['author'] = author.text or ""
        
        # Date
        updated = entry.find('{http://www.w3.org/2005/Atom}updated')
        if updated is not None:
            data['date'] = updated.text or ""
        
        return data
    
    def collect_all_feeds(self, feed_configs: List[FeedConfig]) -> dict:
        """Collect articles from all configured feeds."""
        stats = {
            "total_fetched": 0,
            "total_added": 0,
            "feeds_processed": 0,
            "ai_relevant_added": 0
        }
        
        for feed_config in feed_configs:
            if not feed_config.enabled:
                print(f"Skipping disabled feed: {feed_config.name}")
                continue
            
            print(f"Processing feed: {feed_config.name}")
            articles = self.fetch_feed(feed_config, max_articles=50)
            
            added_count = 0
            ai_count = 0
            
            for article in articles:
                if self.database.add_article(article):
                    added_count += 1
                    if article.ai_relevant:
                        ai_count += 1
            
            stats["total_fetched"] += len(articles)
            stats["total_added"] += added_count
            stats["ai_relevant_added"] += ai_count
            stats["feeds_processed"] += 1
            
            print(f"  Added: {added_count}/{len(articles)} articles, AI-relevant: {ai_count}")
            
            # Be respectful to servers
            time.sleep(1)
        
        return stats