"""Simple news collection module for RSS feeds using only standard library."""

import urllib.request
import urllib.parse
from datetime import datetime
from typing import List, Dict, Any
import time
import re
import html

from .config import FeedConfig, Config, RegionConfig
from .database import Article, Database
from .security_utils import (
    parse_xml_safe, clean_text_content, validate_url, safe_urlopen
)


class SimpleCollector:
    """Collects news from RSS feeds using only standard library."""
    
    def __init__(self, database: Database):
        self.database = database
        self.headers = {
            'User-Agent': 'AI-News-Collector/1.0 (Simple RSS Reader)'
        }
    
    def clean_html(self, html_content: str) -> str:
        """Remove HTML tags using secure sanitization."""
        return clean_text_content(html_content)
    
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
    
    def fetch_rss_feed(self, url: str):
        """Fetch and parse RSS feed securely."""
        try:
            # Validate URL first
            is_valid, error = validate_url(url)
            if not is_valid:
                print(f"URL validation failed for {url}: {error}")
                return None
            
            # Use safe URL opener
            response = safe_urlopen(url, headers=self.headers, timeout=30)
            if response is None:
                return None
            
            with response:
                content = response.read()
            
            # Parse XML securely
            root = parse_xml_safe(content.decode('utf-8', errors='ignore'))
            return root
            
        except Exception as e:
            print(f"Error fetching RSS feed from {url}: {e}")
            return None
    
    def parse_rss_item(self, item) -> dict:
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
        # Handle namespace for content:encoded
        content = None
        if hasattr(item, 'find'):
            # Try content:encoded with namespace
            content = item.find('.//{http://purl.org/rss/1.0/modules/content/}encoded')
            if content is None:
                content = item.find('description')
        
        if content is not None and hasattr(content, 'text'):
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
        
        if root is None:
            return articles
        
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
    
    def parse_atom_entry(self, entry) -> dict:
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

    def collect_region(self, config: Config, region: str) -> Dict[str, Any]:
        """Collect news from specific region only."""
        if region not in config.regions:
            print(f"❌ Unknown region: {region}")
            return {"feeds_processed": 0, "total_fetched": 0, "total_added": 0}
        
        region_config = config.regions[region]
        if not region_config.enabled:
            print(f"⚠️  Region {region} is disabled")
            return {"feeds_processed": 0, "total_fetched": 0, "total_added": 0}
        
        print(f"🌍 Collecting news from {region_config.name} ({region.upper()})...")
        
        stats = {
            "feeds_processed": 0,
            "total_fetched": 0,
            "total_added": 0,
            "ai_relevant_added": 0
        }
        
        for feed in region_config.feeds:
            if not feed.enabled:
                continue
                
            print(f"  📡 Processing {feed.name}...")
            feed_stats = self._process_feed(feed, region)
            
            stats["feeds_processed"] += 1
            stats["total_fetched"] += feed_stats["fetched"]
            stats["total_added"] += feed_stats["added"]
            stats["ai_relevant_added"] += feed_stats["ai_relevant"]
        
        print(f"✅ {region_config.name} collection complete:")
        print(f"   Feeds processed: {stats['feeds_processed']}")
        print(f"   Articles fetched: {stats['total_fetched']}")
        print(f"   Articles added: {stats['total_added']}")
        print(f"   AI-relevant added: {stats['ai_relevant_added']}")
        
        return stats

    def collect_multiple_regions(self, config: Config, regions: List[str]) -> Dict[str, Any]:
        """Collect news from multiple regions."""
        total_stats: Dict[str, Any] = {
            "regions_processed": 0,
            "feeds_processed": 0,
            "total_fetched": 0,
            "total_added": 0,
            "ai_relevant_added": 0,
            "region_stats": {}
        }
        
        for region in regions:
            if region in config.regions and config.regions[region].enabled:
                region_stats = self.collect_region(config, region)
                total_stats["regions_processed"] = total_stats["regions_processed"] + 1
                total_stats["feeds_processed"] = total_stats["feeds_processed"] + region_stats["feeds_processed"]
                total_stats["total_fetched"] = total_stats["total_fetched"] + region_stats["total_fetched"]
                total_stats["total_added"] = total_stats["total_added"] + region_stats["total_added"]
                total_stats["ai_relevant_added"] = total_stats["ai_relevant_added"] + region_stats["ai_relevant_added"]
                total_stats["region_stats"][region] = region_stats
        
        return total_stats

    def _process_feed(self, feed: FeedConfig, region: str = "global") -> Dict[str, Any]:
        """Process a single feed and save articles."""
        try:
            articles = self.fetch_feed(feed)
            
            stats = {"fetched": len(articles), "added": 0, "ai_relevant": 0}
            
            for article in articles:
                # Update article with region
                article.region = region
                
                if self.database.add_article(article):
                    stats["added"] += 1
                    if article.ai_relevant:
                        stats["ai_relevant"] += 1
            
            return stats
            
        except Exception as e:
            print(f"❌ Error processing {feed.name}: {e}")
            return {"fetched": 0, "added": 0, "ai_relevant": 0}