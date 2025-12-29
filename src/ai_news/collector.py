"""Simple news collection module for RSS feeds using only standard library."""

import urllib.request
import urllib.parse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
import time
import re
import html
from difflib import SequenceMatcher
from dataclasses import dataclass
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np

from .config import FeedConfig, Config, RegionConfig, get_performance_config
from .database import Article, Database
from .performance_metrics import get_metrics
from .security_utils import (
    parse_xml_safe, clean_text_content, validate_url, safe_urlopen
)


class SimpleCollector:
    """Collects news from RSS feeds using only standard library."""

    def __init__(self, database: Database, max_workers: int = 5):
        self.database = database
        self.metrics = get_metrics()
        self.perf_config = get_performance_config()
        self.max_workers = max_workers  # Parallel feed fetching
        self.headers = {
            'User-Agent': 'AI-News-Collector/1.0 (Simple RSS Reader)'
        }
        self._lock = threading.Lock()  # Thread-safe database writes
        self._semantic_model = None  # Lazy loaded FastEmbed model

    def _get_semantic_model(self):
        """Lazy load FastEmbed model for semantic matching."""
        if self._semantic_model is None:
            try:
                from fastembed import TextEmbedding
                self._semantic_model = TextEmbedding()
            except ImportError:
                raise ImportError("FastEmbed not installed. Install with: pip install fastembed")
        return self._semantic_model

    def _is_ai_relevant(self, article: Article) -> bool:
        """Simple AI relevance check using keywords."""
        CORE_AI_KEYWORDS = {
            # Core AI/ML terms
            'machine learning', 'deep learning', 'neural network', 'artificial intelligence', 'ai',
            'gpt', 'llm', 'large language model', 'transformer', 'diffusion model',
            'computer vision', 'natural language processing', 'nlp', 'reinforcement learning',
            'generative ai', 'chatgpt', 'openai', 'anthropic', 'claude', 'gemini',

            # Extended AI companies/organizations
            'hugging face', 'stability ai', 'midjourney', 'deepmind', 'google deepmind',
            'microsoft ai', 'meta ai', 'nvidia', 'groq', 'mistral ai',

            # Additional AI technologies
            'embedding', 'vector database', 'rag', 'retrieval augmented generation',
            'prompt engineering', 'fine-tuning', 'foundation model', 'multimodal',

            # AI applications
            'text generation', 'image generation', 'code generation', 'ai assistant'
        }
        text = f"{article.title or ''} {article.summary or ''} {article.content or ''}".lower()
        return any(kw in text for kw in CORE_AI_KEYWORDS)

    def should_collect_article(self, article: Article, topics: Optional[List[str]] = None) -> bool:
        """Check if article should be collected based on semantic similarity to topics.

        Args:
            article: Article to check
            topics: List of topics to match against (None or empty = collect all)

        Returns:
            True if article should be collected, False otherwise
        """
        # Default behavior: collect all articles if no topics specified
        if not topics:
            return True

        try:
            model = self._get_semantic_model()

            # Prepare article text
            article_text = f"{article.title}. {article.summary or article.content or ''}"
            article_text = article_text[:2000]  # Truncate if too long

            # Generate article embedding
            article_embedding = list(model.embed([article_text]))[0]

            # Check similarity against each topic (OR logic - match ANY topic)
            for topic in topics:
                # Generate topic embedding
                topic_embedding = list(model.embed([topic]))[0]

                # Calculate cosine similarity
                similarity = np.dot(topic_embedding, article_embedding) / (
                    np.linalg.norm(topic_embedding) * np.linalg.norm(article_embedding)
                )

                # If similarity meets threshold for ANY topic, collect it
                if similarity >= 0.55:
                    return True

            # No topic matched
            return False

        except ImportError:
            # FastEmbed not installed - collect all articles (graceful degradation)
            return True
        except Exception as e:
            # Log error but don't fail collection
            print(f"Warning: Semantic filtering error: {e}. Collecting article.")
            return True

    @staticmethod
    def _check_http_status(response, url: str) -> bool:
        """Check HTTP response status and return True if OK.

        Prints warning and returns False for error status codes.
        """
        try:
            if hasattr(response, 'status'):
                status = response.status
            else:
                status = response.getcode()
        except Exception:
            return True

        if status == 403:
            print(f"⚠ Access forbidden (403) from {url}")
            return False
        elif status == 404:
            print(f"⚠ Feed not found (404) from {url}")
            return False
        elif status >= 400:
            print(f"⚠ Server error ({status}) from {url}")
            return False

        return True

    def fetch_rss_feed(self, url: str):

        """Fetch and parse RSS feed securely."""
        try:
            # Validate URL first
            is_valid, error = validate_url(url)
            if not is_valid:
                print(f"URL validation failed for {url}: {error}")
                return None

            # Use safe URL opener (reduced timeout for faster failure on slow feeds)
            response = safe_urlopen(url, headers=self.headers, timeout=10)
            if response is None:
                return None

            # Check HTTP status
            if not self._check_http_status(response, url):
                return None

            # Read content (response is guaranteed not-None here)
            try:
                content_bytes = response.read()  # type: ignore
                content = content_bytes.decode('utf-8', errors='ignore')
            finally:
                try:
                    response.close()  # type: ignore
                except:
                    pass

            # Check if response is HTML (blocking page) instead of XML/RSS
            # More lenient check: reject ONLY if it's clearly an HTML document
            # Valid RSS/Atom feeds start with <?xml or <rss or <feed or <atom
            content_lower = content.lower().strip()
            if content_lower.startswith('<!doctype html>') or (content_lower.startswith('<html>') and not content_lower.startswith('<?xml')):
                print(f"⚠ Received HTML instead of RSS from {url} (likely blocked or 403)")
                return None

            # Parse XML securely
            root = parse_xml_safe(content, source_url=url)
            return root

        except Exception as e:
            error_str = str(e)
            if '403' in error_str or 'Forbidden' in error_str:
                print(f"⚠ Access forbidden (403) from {url}")
            elif '404' in error_str or 'Not Found' in error_str:
                print(f"⚠ Feed not found (404) from {url}")
            else:
                print(f"Error fetching RSS feed from {url}: {e}")
            return None

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
                    region="global",  # Default region, will be updated in _process_feed
                    ai_relevant=False,  # Will be set by confidence scorer
                    ai_keywords_found=[]
                )

                ai_relevant = self._is_ai_relevant(article)
                article.ai_relevant = ai_relevant
                article.ai_confidence = 0.8 if ai_relevant else 0.0

                if ai_relevant:
                    articles.append(article)
                
            except Exception as e:
                print(f"    Error processing article: {e}")
                continue
        
        print(f"  Found {len(articles)} AI-relevant articles (confidence >= 0.7)")
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
    
    def collect_all_feeds(self, feed_configs: List[FeedConfig], max_articles_per_feed: int = 25, topics: Optional[List[str]] = None) -> dict:
        """Collect articles from all configured feeds.

        Args:
            feed_configs: List of feed configurations
            max_articles_per_feed: Maximum articles to collect per feed
            topics: Optional list of topics for semantic filtering

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_fetched": 0,
            "total_added": 0,
            "feeds_processed": 0,
            "ai_relevant_added": 0,
            "semantic_filtered": 0
        }

        for feed_config in feed_configs:
            if not feed_config.enabled:
                print(f"Skipping disabled feed: {feed_config.name}")
                continue

            print(f"Processing feed: {feed_config.name}")
            articles = self.fetch_feed(feed_config, max_articles=max_articles_per_feed)

            # Apply semantic filtering if topics provided
            if topics is not None:
                filtered_count = 0
                filtered_articles = []
                for article in articles:
                    if self.should_collect_article(article, topics):
                        filtered_articles.append(article)
                    else:
                        filtered_count += 1
                articles = filtered_articles
                stats["semantic_filtered"] += filtered_count

                if filtered_count > 0:
                    print(f"  Semantic filter: {filtered_count}/{len(articles) + filtered_count} articles filtered")

            added_count = 0
            ai_count = 0

            for article in articles:
                if self.database.save_article(article, skip_entities=True):
                    added_count += 1
                    if article.ai_relevant:
                        ai_count += 1

            stats["total_fetched"] += len(articles)
            stats["total_added"] += added_count
            stats["ai_relevant_added"] += ai_count
            stats["feeds_processed"] += 1

            print(f"  Added: {added_count}/{len(articles)} articles, AI-relevant: {ai_count}")

            # Be respectful to servers (removed sleep - too slow for 15 feeds)

        # Log performance summary at end
        self.metrics.log_summary()

        return stats

    def collect_region(self, config: Config, region: str, topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Collect news from specific region only.

        Args:
            config: Configuration object
            region: Region name to collect
            topics: Optional list of topics for semantic filtering

        Returns:
            Statistics dictionary
        """
        if region not in config.regions:
            print(f"❌ Unknown region: {region}")
            return {"feeds_processed": 0, "total_fetched": 0, "total_added": 0, "ai_relevant_added": 0}

        region_config = config.regions[region]
        if not region_config.enabled:
            print(f"⚠️  Region {region} is disabled")
            return {"feeds_processed": 0, "total_fetched": 0, "total_added": 0, "ai_relevant_added": 0}

        print(f"🌍 Collecting news from {region_config.name} ({region.upper()}):", flush=True)

        stats = {
            "feeds_processed": 0,
            "total_fetched": 0,
            "total_added": 0,
            "ai_relevant_added": 0,
            "semantic_filtered": 0
        }

        for feed in region_config.feeds:
            if not feed.enabled:
                continue

            print(f"  📡 Processing {feed.name}...", flush=True)
            feed_stats = self._process_feed(feed, region, topics=topics)

            stats["feeds_processed"] += 1
            stats["total_fetched"] += feed_stats["fetched"]
            stats["total_added"] += feed_stats["added"]
            stats["ai_relevant_added"] += feed_stats["ai_relevant"]
            stats["semantic_filtered"] += feed_stats.get("semantic_filtered", 0)

        print(f"✅ {region_config.name} collection complete:", flush=True)
        print(f"   Feeds processed: {stats['feeds_processed']}", flush=True)
        print(f"   Articles fetched: {stats['total_fetched']}", flush=True)
        print(f"   Articles added: {stats['total_added']}", flush=True)
        print(f"   AI-relevant added: {stats['ai_relevant_added']}", flush=True)
        if stats["semantic_filtered"] > 0:
            print(f"   Semantically filtered: {stats['semantic_filtered']}", flush=True)

        return stats

    def collect_multiple_regions(self, config: Config, regions: List[str], parallel: bool = True, topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Collect news from multiple regions with optimized parallel processing.

        Performance optimizations:
        - Parallel feed fetching (network I/O bound, max_workers concurrent connections)
        - Batch database writes (single transaction to avoid lock contention)
        - Thread-safe operations with locks

        Args:
            config: Configuration object
            regions: List of region names to collect
            parallel: Use parallel feed fetching (default: True)
            topics: Optional list of topics for semantic filtering

        Returns:
            Statistics dictionary with keys: feeds_processed, total_fetched, total_added, ai_relevant_added
        """
        total_stats: Dict[str, Any] = {
            "regions_processed": 0,
            "feeds_processed": 0,
            "total_fetched": 0,
            "total_added": 0,
            "ai_relevant_added": 0,
            "region_stats": {}
        }

        # Collect all feeds from all regions
        all_feeds = []
        for region in regions:
            if region in config.regions and config.regions[region].enabled:
                region_config = config.regions[region]
                for feed in region_config.feeds:
                    if feed.enabled:
                        all_feeds.append((feed, region))

        if not all_feeds:
            return total_stats

        if parallel and len(all_feeds) > 1:
            # Parallel: Fetch feeds concurrently, batch DB writes
            all_articles = []

            # Phase 1: Parallel fetch (network I/O bound)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_feed = {
                    executor.submit(self.fetch_feed, feed): (feed, region)
                    for feed, region in all_feeds
                }

                for future in as_completed(future_to_feed):
                    feed, region = future_to_feed[future]
                    try:
                        articles = future.result()

                        # Apply semantic filtering if topics provided
                        if topics is not None:
                            articles = [a for a in articles if self.should_collect_article(a, topics)]

                        # Add region to articles
                        for article in articles:
                            article.region = region
                        all_articles.extend(articles)

                        with self._lock:
                            total_stats["feeds_processed"] = int(total_stats.get("feeds_processed", 0)) + 1  # type: ignore
                            total_stats["total_fetched"] = int(total_stats.get("total_fetched", 0)) + len(articles)  # type: ignore
                    except Exception as e:
                        print(f"❌ Error fetching {feed.name}: {e}")

            # Phase 2: Batch database writes (avoid lock contention)
            print(f"💾 Saving {len(all_articles)} articles to database...")
            added_count = 0
            ai_count = 0

            for article in all_articles:
                if self.database.save_article(article, skip_entities=True):
                    added_count += 1
                    if article.ai_relevant:
                        ai_count += 1

            total_stats["total_added"] = added_count  # type: ignore
            total_stats["ai_relevant_added"] = ai_count  # type: ignore
        else:
            # Sequential processing (original behavior)
            for feed, region in all_feeds:
                feed_stats = self._process_feed(feed, region, topics=topics)
                total_stats["feeds_processed"] = int(total_stats.get("feeds_processed", 0)) + 1  # type: ignore
                total_stats["total_fetched"] = int(total_stats.get("total_fetched", 0)) + feed_stats.get("fetched", 0)  # type: ignore
                total_stats["total_added"] = int(total_stats.get("total_added", 0)) + feed_stats.get("added", 0)  # type: ignore
                total_stats["ai_relevant_added"] = int(total_stats.get("ai_relevant_added", 0)) + feed_stats.get("ai_relevant", 0)  # type: ignore

        total_stats["regions_processed"] = len(set(r for f, r in all_feeds))  # type: ignore
        return total_stats

    def _process_feed(self, feed: FeedConfig, region: str = "global", topics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process a single feed and save articles.

        Args:
            feed: Feed configuration
            region: Region for articles
            topics: Optional list of topics for semantic filtering

        Returns:
            Statistics dictionary
        """
        try:
            articles = self.fetch_feed(feed)

            stats = {
                "fetched": len(articles),
                "added": 0,
                "ai_relevant": 0,
                "semantic_filtered": 0
            }

            for article in articles:
                start = time.time()
                # Update article with region
                article.region = region

                # Apply semantic filtering if topics provided
                if topics is not None:
                    if not self.should_collect_article(article, topics):
                        stats["semantic_filtered"] += 1
                        continue

                # Skip entity extraction during collection for faster performance
                if self.database.save_article(article, skip_entities=True):
                    stats["added"] += 1
                    if article.ai_relevant:
                        stats["ai_relevant"] += 1

            # Log semantic filtering stats if applicable
            if topics is not None and stats["semantic_filtered"] > 0:
                print(f"  Semantic filter: {stats['semantic_filtered']}/{stats['fetched']} articles filtered")
                topics_str = ', '.join(topics)
                print(f"  Topics: {topics_str}")

            return stats

        except Exception as e:
            print(f"❌ Error processing {feed.name}: {e}")
            return {"fetched": 0, "added": 0, "ai_relevant": 0, "semantic_filtered": 0}
