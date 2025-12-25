"""Feed Discovery System for AI News.

This module provides automatic discovery of RSS feeds for specified topics
using web search capabilities and feed validation.
"""

import re
import json
import sqlite3
import requests
import feedparser
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from urllib.parse import urljoin, urlparse

from .search_collector import SearchEngineCollector
from .database import Database
from .feed_cache import FeedCache
from urllib.parse import unquote

logger = logging.getLogger(__name__)


class FeedDiscoveryError(Exception):
    """Base exception for feed discovery errors."""
    pass


class FeedValidationError(Exception):
    """Raised when feed validation fails."""
    pass


class NoFeedsFoundError(Exception):
    """Raised when no feeds are found for a topic."""
    pass


class FeedDiscoveryEngine:
    """Discover RSS feeds for topics using web search."""

    def __init__(self, database=None):
        if database:
            self.web_search = SearchEngineCollector(database)
            self.cache = FeedCache(database)
        else:
            # For discovery without database, use a minimal approach
            self.web_search = None
            self.cache = None

        # Initialize RSS patterns for URL extraction
        self._rss_patterns = [
            r'https?://[^\s]+/rss(?:/[^\s]*)?',
            r'https?://[^\s]+/feed(?:/[^\s]*)?',
            r'https?://[^\s]+\.xml',
            r'https?://[^\s]+/atom\.xml',
            r'https?://[^\s]+/rss\.xml'
        ]

    def decompose_topic(self, topic: str) -> List[str]:
        """Decompose combined topics into component topics."""
        topic_lower = topic.lower()
        components = []

        # Pattern 1: "X in Y" -> [X, Y]
        if ' in ' in topic_lower:
            parts = topic_lower.split(' in ', 1)
            components.extend([parts[0].strip(), parts[1].strip()])

        # Pattern 2: "X for Y" -> [X, Y]
        elif ' for ' in topic_lower:
            parts = topic_lower.split(' for ', 1)
            components.extend([parts[0].strip(), parts[1].strip()])

        # Pattern 3: "X and Y" -> [X, Y]
        elif ' and ' in topic_lower:
            parts = topic_lower.split(' and ', 1)
            components.extend([parts[0].strip(), parts[1].strip()])

        # Pattern 4: "X + Y" (plus sign separator) -> [X, Y]
        elif ' + ' in topic_lower:
            parts = topic_lower.split(' + ', 1)
            components.extend([parts[0].strip(), parts[1].strip()])

        # Pattern 5: "X+Y" (no space around plus) -> [X, Y]
        elif '+' in topic_lower:
            parts = topic_lower.split('+', 1)
            components.extend([parts[0].strip(), parts[1].strip()])

        # If no patterns matched, return original
        if not components:
            components = [topic_lower]

        # Remove duplicates and return
        return list(set(components))

    def _score_by_intersection(self, feed_text: str, components: List[str]) -> float:
        """
        Score feed by how many topic components it covers.

        Args:
            feed_text: Feed title + description + URL (lowercase)
            components: List of topic components (e.g., ['AI', 'healthcare'])

        Returns:
            Score from 0.0 to 1.0 indicating component coverage
        """
        if not components:
            return 0.0

        feed_text_lower = feed_text.lower()
        components_mentioned = 0

        for component in components:
            if component.lower() in feed_text_lower:
                components_mentioned += 1

        return components_mentioned / len(components)

    def _get_cached_terms(self, cache_key: str) -> Optional[List[str]]:
        """Retrieve cached terms from discovered_feeds table."""
        try:
            # Get database path from cache if available
            if not self.cache or not hasattr(self.cache, 'db_path'):
                return None

            db_path = self.cache.db_path if hasattr(self.cache, 'db_path') else None
            if not db_path:
                return None

            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT title FROM discovered_feeds
                    WHERE topic = ? AND feed_url = '__terms_cache__'
                """, (cache_key,))
                rows = cursor.fetchall()

                if rows:
                    # Parse JSON array from title field
                    return json.loads(rows[0][0])
        except Exception as e:
            logger.debug(f"Cache lookup failed: {e}")
        return None

    def _cache_terms(self, cache_key: str, terms: List[str], original_topic: str):
        """Cache discovered terms in discovered_feeds table."""
        try:
            # Get database path from cache if available
            if not self.cache or not hasattr(self.cache, 'db_path'):
                return

            db_path = self.cache.db_path if hasattr(self.cache, 'db_path') else None
            if not db_path:
                return

            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO discovered_feeds
                    (topic, feed_url, title, description, validated, last_updated)
                    VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
                """, (cache_key, '__terms_cache__', json.dumps(terms), f'Terms for {original_topic}'))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to cache terms: {e}")

    def _extract_terms_from_search(self, search_results: List[Dict], components: List[str]) -> List[str]:
        """Extract industry terms from search results using heuristics.

        Args:
            search_results: List of search result dicts with 'title' and 'snippet'
            components: Original topic components for filtering

        Returns:
            List of extracted terms
        """
        terms = []
        component_set = {c.lower() for c in components}
        exclude_words = {'ai', 'artificial', 'intelligence', 'technology', 'tech',
                         'industry', 'sector', 'field', 'area', 'uses', 'used',
                         'application', 'applications', 'based', 'about', 'what'}

        for result in search_results[:5]:  # Check top 5 results
            title = result.get('title', '')
            snippet = result.get('content', '') or result.get('snippet', '')
            text = f"{title} {snippet}".lower()

            # Pattern 1: "X is called Y" or "X, also known as Y"
            also_known = re.search(
                r'(?:also known as|called|referred to as|often called|known as)\s+["\']?([a-z]{4,15})["\']?',
                text
            )
            if also_known:
                term = also_known.group(1)
                if term not in component_set and term not in exclude_words:
                    terms.append(term)

            # Pattern 2: Single compound words ending in tech/finance/care/etc (industry terms)
            compounds = re.findall(r'\b([a-z]{4,15}(?:tech|finance|care|chain|quantum|surance))\b', text)
            for compound in compounds:
                if compound not in component_set and compound not in exclude_words:
                    terms.append(compound)

            # Pattern 3: Capitalized terms in snippets (industry names)
            capitals = re.findall(r'\b([A-Z][a-z]+(?:tech|finance|Tech|AI|Health|Insur))\b', snippet)
            for capital in capitals:
                term = capital.lower()
                if term not in component_set and term not in exclude_words:
                    terms.append(term)

        return terms

    def _discover_intersection_terms(self, components: List[str], original_topic: str) -> List[str]:
        """Discover industry-specific terminology for intersection topics.

        Uses web search to find industry terms like "insurtech" for "insurance + ai".
        Results are cached in the discovered_feeds table for reuse.

        Args:
            components: Decomposed topic components (e.g., ["insurance", "ai"])
            original_topic: Original topic string (e.g., "insurance + ai")

        Returns:
            List of discovered terms (e.g., ["insurtech", "insurance technology"])
        """
        cache_key = f"{original_topic}_terms"
        logger.info(f"Discovering intersection terms for: {original_topic}")

        # Check cache first
        cached_terms = self._get_cached_terms(cache_key)
        if cached_terms:
            logger.info(f"Using cached terms: {cached_terms}")
            return cached_terms

        # Generate search queries for terminology discovery
        term_queries = [
            f"{components[0]} {components[1]} industry term",
            f"{components[0]} {components[1]} technology",
            f"{components[0]} {components[1]} called",
        ]

        discovered_terms = []

        for query in term_queries:
            try:
                results = self.web_search.search_searxng(query, max_results=10)
                if results:
                    extracted = self._extract_terms_from_search(results, components)
                    discovered_terms.extend(extracted)

                    if discovered_terms:
                        break  # Found terms, no need to search more
            except Exception as e:
                logger.warning(f"Term search failed for '{query}': {e}")
                continue

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = [t for t in discovered_terms if not (t in seen or seen.add(t))]

        if unique_terms:
            self._cache_terms(cache_key, unique_terms, original_topic)
            logger.info(f"Discovered and cached terms: {unique_terms}")
        else:
            logger.warning(f"No terms discovered for {original_topic}, will use components")

        return unique_terms

    def discover_feeds_for_topic(self, topic: str, max_feeds: int = 5, force_discovery: bool = False) -> List[Dict[str, Any]]:
        """
        Discover RSS feeds for a specified topic.

        Args:
            topic: Target topic for feed discovery
            max_feeds: Maximum number of feeds to return
            force_discovery: Skip cache and force web search

        Returns:
            List of dictionaries containing feed information
        """
        # Validate input
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty or whitespace")

        logger.info(f"Discovering feeds for topic: {topic}")

        # Check cache first (unless forcing discovery)
        if not force_discovery and self.cache:
            cached_feeds = self.cache.check_cache(topic)
            if cached_feeds:
                logger.info("Using cached results (instant)")
                return cached_feeds[:max_feeds]

        # Check if this is a combined topic
        components = self.decompose_topic(topic)

        if len(components) > 1:
            logger.info(f"Detected combined topic: {topic} -> {components}")
            feeds = self._discover_intersection_feeds(components, topic, max_feeds)
        else:
            feeds = self._discover_single_topic_feeds(topic, max_feeds)

        # Save to cache for future instant access
        if feeds and self.cache:
            self.cache.save_to_cache(topic, feeds)

        return feeds

    def _discover_single_topic_feeds(self, topic: str, max_feeds: int) -> List[Dict[str, Any]]:
        """Discover feeds for a single topic."""
        logger.info(f"Discovering feeds for single topic: {topic}")

        # Generate search queries
        search_queries = self._generate_search_queries(topic)

        # Search for RSS feeds using web search
        all_discovered = set()

        # Web search with aggressive scraping
        if self.web_search:
            for i, query in enumerate(search_queries[:5]):  # Try up to 5 queries (increased from 3)
                try:
                    logger.info(f"Search {i+1}: {query}")
                    search_results = self.web_search.search_searxng(query, max_results=5)

                    # FALLBACK: Try DuckDuckGo if SearXNG fails
                    if not search_results or len(search_results) < 2:
                        logger.info(f"  SearXNG returned {len(search_results) if search_results else 0} results, trying DuckDuckGo")
                        search_results = self.web_search.search_duckduckgo(query, max_results=5)
                        if search_results:
                            logger.info(f"  DuckDuckGo returned {len(search_results)} results")

                    if not search_results:
                        logger.info("  No results from search")
                        continue

                    if search_results is None:
                        logger.warning("  Search returned None - search service may be unavailable")
                        continue

                    for result in search_results[:3]:  # Only check top 3 results
                        if not result or not isinstance(result, dict):
                            logger.warning(f"  Invalid search result format: {result}")
                            continue

                        result_url = result.get('url', '')
                        title = result.get('title', '')
                        content = result.get('content', '')

                        # Unwrap DuckDuckGo redirect URLs
                        unwrapped_url = self._unwrap_redirect_url(result_url)
                        if unwrapped_url != result_url:
                            logger.debug(f"  Unwrapped redirect URL: {unwrapped_url[:60]}")

                        # Extract RSS URLs from search result text
                        search_text = f"{title} {content} {unwrapped_url}"
                        extracted = self._extract_rss_urls(search_text)
                        all_discovered.update(extracted)

                        # Only scrape if it's a high-priority feed directory
                        if self._is_promising_feed_directory(unwrapped_url, title, content):
                            logger.info(f"  Scraping directory: {title}")
                            scraped_feeds = self._discover_feeds_from_page(unwrapped_url)
                            all_discovered.update(scraped_feeds[:5])  # Limit scraped feeds

                        # If URL itself looks like RSS, add it
                        if unwrapped_url and self._looks_like_rss_url(unwrapped_url):
                            all_discovered.add(unwrapped_url)

                    # Early exit if we have enough feeds
                    if len(all_discovered) >= 5:
                        logger.info(f"Found {len(all_discovered)} feeds, stopping search")
                        break

                except Exception as e:
                    logger.warning(f"Search failed for query '{query}': {e}")
                    continue

        # FALLBACK: If no feeds found, try broader category searches
        if not all_discovered and self.web_search:
            logger.warning("Primary search found no feeds, trying fallback patterns")
            fallback_queries = self._generate_fallback_queries(topic)

            for query in fallback_queries[:3]:
                try:
                    logger.info(f"Fallback search: {query}")
                    search_results = self.web_search.search_searxng(query, max_results=5)

                    if not search_results:
                        search_results = self.web_search.search_duckduckgo(query, max_results=5)

                    if search_results:
                        for result in search_results[:3]:
                            result_url = result.get('url', '')
                            title = result.get('title', '')
                            content = result.get('content', '')

                            unwrapped_url = self._unwrap_redirect_url(result_url)
                            search_text = f"{title} {content} {unwrapped_url}"
                            extracted = self._extract_rss_urls(search_text)
                            all_discovered.update(extracted)

                            if self._is_promising_feed_directory(unwrapped_url, title, content):
                                scraped_feeds = self._discover_feeds_from_page(unwrapped_url)
                                all_discovered.update(scraped_feeds[:3])

                            if unwrapped_url and self._looks_like_rss_url(unwrapped_url):
                                all_discovered.add(unwrapped_url)

                        if len(all_discovered) >= 3:
                            logger.info(f"Fallback found {len(all_discovered)} feeds")
                            break

                except Exception as e:
                    logger.debug(f"Fallback search failed for '{query}': {e}")
                    continue

        logger.info(f"Total RSS candidates discovered: {len(all_discovered)}")

        if not all_discovered:
            raise NoFeedsFoundError(f"No RSS feeds found for topic: {topic}")

        # Validate and return good feeds
        valid_feeds = []
        for rss_url in list(all_discovered)[:max_feeds * 2]:  # Check more feeds than needed
            try:
                feed_info = self._validate_feed(rss_url, topic)
                if feed_info:
                    valid_feeds.append(feed_info)
                    if len(valid_feeds) >= max_feeds:
                        break

            except Exception as e:
                logger.warning(f"Feed validation failed for {rss_url}: {e}")
                continue

        if not valid_feeds:
            raise NoFeedsFoundError(f"No valid RSS feeds found for topic: {topic}")

        # Sort by relevance score
        valid_feeds.sort(key=lambda x: x['relevance_score'], reverse=True)

        logger.info(f"Successfully validated {len(valid_feeds)} feeds for topic: {topic}")
        return valid_feeds[:max_feeds]

    def _discover_intersection_feeds(self, components: List[str], original_topic: str, max_feeds: int) -> List[Dict[str, Any]]:
        """Discover feeds for intersection of multiple topics.

        Args:
            components: List of topic components (e.g., ['AI', 'healthcare'])
            original_topic: Original user input (e.g., 'AI in healthcare')
            max_feeds: Maximum feeds to return

        Returns:
            List of feeds covering the intersection
        """
        logger.info(f"Discovering intersection feeds for: {components}")

        all_feeds = set()

        # Smart terminology discovery - find industry terms like "insurtech" for "insurance + ai"
        discovered_terms = self._discover_intersection_terms(components, original_topic)

        # Generate search queries using discovered terms or fallback to component combinations
        search_queries = []

        if discovered_terms:
            # Use discovered terms for primary searches (higher quality feeds)
            for term in discovered_terms[:2]:  # Use top 2 discovered terms
                search_queries.extend([
                    f'"{term}" RSS feed',
                    f'"{term}" blog RSS',
                    f'"{term}" news feed',
                ])
            logger.info(f"Using discovered terms for queries: {discovered_terms[:2]}")
        else:
            # Fallback: Use component combinations
            search_queries = [
                f'"{original_topic}" RSS feed',
                f'"{components[0]} {components[1]}" news RSS',
                f'"{components[0]}" "{components[1]}" RSS feed',
            ]

        # Search for feeds using the generated queries
        if self.web_search:
            for query in search_queries[:5]:  # Try up to 5 queries
                try:
                    search_results = self.web_search.search_searxng(query, max_results=5)
                    for result in search_results[:3]:
                        url = result.get('url', '')
                        # Direct RSS URLs
                        if self._looks_like_rss_url(url):
                            all_feeds.add(url)
                        # Scrape promising pages for more feeds
                        elif self._is_promising_feed_directory(url, result.get('title', ''), result.get('content', '')):
                            scraped = self._discover_feeds_from_page(url)
                            all_feeds.update(scraped[:10])  # Get more feeds from directories
                except Exception as e:
                    logger.debug(f"  Search failed for '{query}': {e}")

        logger.info(f"Total intersection candidates: {len(all_feeds)}")

        # Filter out all directory/aggregator URLs - only keep real content feeds
        all_feeds = {
            url for url in all_feeds
            if not any(domain in url.lower() for domain in ['feedspot.com', 'rss-feeds.org'])
        }

        logger.info(f"After filtering: {len(all_feeds)} valid feed candidates")

        if not all_feeds:
            raise NoFeedsFoundError(f"No RSS feeds found for intersection: {original_topic}")

        # Validate and score feeds by intersection relevance
        valid_feeds = []

        for rss_url in list(all_feeds)[:max_feeds * 5]:  # Check MORE feeds for intersections
            try:
                # For intersection topics, validate against the first component
                # This is more lenient than requiring the exact original_topic phrase
                validation_topic = components[0] if components else original_topic

                feed_info = self._validate_feed(rss_url, validation_topic)

                if feed_info:
                    # Score by how many components this feed covers
                    feed_text = f"{rss_url} {feed_info.get('title', '')} {feed_info.get('description', '')}".lower()
                    intersection_score = self._score_by_intersection(feed_text, components)

                    feed_info['intersection_score'] = intersection_score
                    feed_info['validated'] = True
                    valid_feeds.append(feed_info)
                else:
                    # Validation returned None - try validating with other components
                    validated = False
                    for component in components[1:]:
                        feed_info = self._validate_feed(rss_url, component)
                        if feed_info:
                            feed_text = f"{rss_url} {feed_info.get('title', '')} {feed_info.get('description', '')}".lower()
                            intersection_score = self._score_by_intersection(feed_text, components)
                            feed_info['intersection_score'] = intersection_score
                            feed_info['validated'] = True
                            valid_feeds.append(feed_info)
                            validated = True
                            break

                    if not validated:
                        raise Exception("Validation returned None for all components")

            except Exception as e:
                # Skip feeds that fail validation - only keep working feeds
                logger.debug(f"Skipping {rss_url}: {str(e)[:100]}")
                continue

        # Sort by intersection score (feeds covering more components first)
        valid_feeds.sort(key=lambda x: x.get('intersection_score', 0), reverse=True)

        logger.info(f"Successfully validated {len(valid_feeds)} intersection feeds")
        return valid_feeds[:max_feeds]

    def _looks_like_rss_url(self, url: str) -> bool:
        """Quick check if URL looks like an RSS feed (not a feed directory)."""
        # Feed directories that look like RSS but should be scraped instead
        directory_patterns = ['feedspot.com/', '/rss_feeds/', '/feed_directory/']

        url_lower = url.lower()
        if any(pattern in url_lower for pattern in directory_patterns):
            return False

        rss_indicators = ['/rss', '/feed', '.xml', '/atom', 'rss.xml', 'feed.xml']
        return any(indicator in url_lower for indicator in rss_indicators)

    def _is_promising_feed_directory(self, url: str, title: str, content: str) -> bool:
        """Check if a search result looks like a feed directory or compilation.
        
        Also returns True for high-quality blogs/companies that likely have RSS feeds.
        """
        if not url:
            return False

        # Check for feed directories/aggregators
        indicators = [
            'rss feeds', 'feed directory', 'rss directory', 'feed list',
            'top rss', 'best rss', 'news aggregator', 'feed compilation',
            'rss feed directory', 'news sources', 'content aggregator'
        ]

        text_to_check = f"{title} {content}".lower()
        score = sum(1 for indicator in indicators if indicator in text_to_check)

        # Also check the URL domain for known feed directories
        feed_domains = ['feedspot.com', 'rss-feeds.org', 'bloglovin.com', 'feedly.com']
        is_feed_domain = any(domain in url.lower() for domain in feed_domains)
        
        if score >= 2 or is_feed_domain:
            return True
        
        # ALSO scrape high-quality blogs/companies (likely to have RSS feeds)
        # Check for blog/company indicators
        blog_indicators = ['blog', 'news', 'quantum', 'microsoft', 'ibm', 'google', 'amazon']
        quality_score = sum(1 for indicator in blog_indicators if indicator in url.lower())
        
        # Check if it's a tech company or reputable domain
        quality_domains = ['microsoft.com', 'ibm.com', 'google.com', 'amazon.com', 
                          'azure.com', 'medium.com', 'substack.com']
        is_quality_domain = any(domain in url.lower() for domain in quality_domains)
        
        # Scrape if it looks like a quality blog/company
        return quality_score >= 1 or is_quality_domain

    def _discover_feeds_from_page(self, url: str) -> List[str]:
        """Visit a promising page and extract RSS feed URLs.

        Enhanced parsing with multiple strategies:
        1. HTML link tags (rel="alternate")
        2. Anchor tags with RSS-indicating paths
        3. JSON-LD structured data
        4. Text-based regex patterns
        """
        feeds = []

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; AI-News-Feed-Discovery/1.0)'
            }

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            content = response.text
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

            # Strategy 1: HTML link tags (most reliable)
            link_patterns = [
                r'<link[^>]*rel=["\']alternate["\'][^>]*type=["\']application/rss\+xml["\'][^>]*href=["\']([^"\']+)["\']',
                r'<link[^>]*rel=["\']alternate["\'][^>]*type=["\']application/atom\+xml["\'][^>]*href=["\']([^"\']+)["\']',
            ]

            for pattern in link_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    full_url = urljoin(base_url, match)
                    feeds.append(full_url)

            # Strategy 2: JSON-LD structured data
            json_ld_pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
            json_ld_matches = re.findall(json_ld_pattern, content, re.IGNORECASE)

            for match in json_ld_matches:
                try:
                    data = json.loads(match)
                    # Look for feed URLs in JSON-LD
                    if isinstance(data, dict):
                        if 'url' in data and any(x in data['url'].lower() for x in ['rss', 'feed', 'atom']):
                            feeds.append(data['url'])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Strategy 3: Anchor tags with RSS paths
            anchor_patterns = [
                r'<a[^>]*href=["\']([^"\']*?/feed[^"\']*)["\']',
                r'<a[^>]*href=["\']([^"\']*?/rss[^"\']*)["\']',
                r'<a[^>]*href=["\']([^"\']*?\.xml)["\']',
            ]

            for pattern in anchor_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    full_url = urljoin(base_url, match)
                    feeds.append(full_url)

            # Strategy 4: Simple text regex (fallback)
            simple_patterns = [
                r'https?://[^\s\'"()<>()]+/feed[^\s\'"()<>()]*',
                r'https?://[^\s\'"()<>()]+/rss[^\s\'"()<>()]*',
                r'https?://[^\s\'"()<>()]+\.xml'
            ]

            for pattern in simple_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                feeds.extend(matches)

        except Exception as e:
            logger.warning(f"Failed to discover feeds from {url}: {e}")

        # Clean and deduplicate
        clean_feeds = []
        seen = set()

        for feed in feeds:
            if feed and feed.startswith('http'):
                # Normalize URL
                feed = feed.split('?')[0].split('#')[0]

                # Verify it looks like a feed
                if any(path in feed.lower() for path in ['/rss', '/feed', '.xml', '/atom']):
                    if feed not in seen:
                        seen.add(feed)
                        clean_feeds.append(feed)

        return clean_feeds

    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain name from URL for display."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            # Clean up common RSS paths
            domain = domain.split('.')[0].title() if '.' in domain else domain.title()
            return domain
        except:
            return 'Feed'

    def _validate_feed(self, rss_url: str, topic: str) -> Optional[Dict[str, Any]]:
        """
        Validate RSS feed and extract basic info.

        Relaxed quality checks (to allow more feeds through):
        - 5+ articles required (relaxed from 10)
        - Last update within 30 days (relaxed from 7)
        - Relevance score >= 0.2 (relaxed from 0.4)
        - Valid RSS/Atom parsing (bozo warnings allowed, errors rejected)

        Returns None if feed fails any quality check.
        """
        try:
            logger.debug(f"Validating feed: {rss_url}")

            headers = {
                'User-Agent': 'AI-News-Feed-Discovery/1.0'
            }

            response = requests.get(rss_url, headers=headers, timeout=15)
            response.raise_for_status()

            # Check content type (lenient)
            content_type = response.headers.get('content-type', '').lower()
            if content_type and not any(ct in content_type for ct in ['xml', 'rss', 'atom', 'json', 'text']):
                logger.debug(f"Skipping {rss_url}: Invalid content type {content_type}")
                return None

            # Parse feed
            feed = feedparser.parse(response.content)

            # Check for critical parsing errors only (allow warnings)
            if feed.bozo and feed.bozo != 0:
                # Only reject on critical exceptions, not warnings
                import feedparser as fp
                if isinstance(feed.bozo_exception, (fp.CharacterEncodingOverride, fp.NonXMLContentType)):
                    # These are warnings, not fatal - log but continue
                    logger.debug(f"Feed has non-fatal parse warning (bozo={feed.bozo}): {rss_url}")
                else:
                    logger.debug(f"Skipping {rss_url}: Critical parse error (bozo={feed.bozo})")
                    return None

            # RELAXED: Must have 5+ entries (was 10)
            if not feed.entries or len(feed.entries) < 5:
                logger.debug(f"Skipping {rss_url}: Insufficient articles ({len(feed.entries) if feed.entries else 0} < 5)")
                return None

            # RELAXED: Last update within 30 days (was 7)
            last_updated = self._get_last_updated(feed)
            if last_updated:
                days_old = (datetime.now() - last_updated).days
                if days_old > 30:
                    logger.debug(f"Skipping {rss_url}: Stale feed ({days_old} days old > 30)")
                    return None

            # Extract feed information
            feed_title = feed.feed.get('title', f'{topic.title()} Feed')
            feed_description = feed.feed.get('description', '')
            article_count = len(feed.entries)

            # DYNAMIC THRESHOLD: More lenient for technical/short topics
            # Technical topics often have varied terminology, so be more flexible
            threshold = 0.15 if len(topic.split()) <= 2 else 0.2
            
            # Calculate relevance score
            relevance_score = self._check_topic_relevance(feed, topic)
            if relevance_score < threshold:
                logger.debug(f"Skipping {rss_url}: Low relevance score {relevance_score} < {threshold}")
                return None

            return {
                'url': rss_url,
                'title': feed_title,
                'description': feed_description,
                'article_count': article_count,
                'relevance_score': relevance_score,
                'last_updated': last_updated,
                'update_frequency': self._estimate_update_frequency(feed),
                'validated': True
            }

        except requests.exceptions.RequestException as e:
            # Enhanced error handling
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                if status_code == 404:
                    logger.warning(f"Feed URL not found (404): {rss_url}")
                elif status_code == 403:
                    logger.warning(f"Access forbidden to feed (403): {rss_url}")
                elif status_code >= 500:
                    logger.warning(f"Server error ({status_code}) for feed: {rss_url}")
                else:
                    logger.debug(f"Network error for {rss_url}: {e}")
            else:
                logger.debug(f"Network error for {rss_url}: {e}")
            return None
        except Exception as e:
            logger.debug(f"Validation error for {rss_url}: {e}")
            return None

    def _get_last_updated(self, feed) -> Optional[datetime]:
        """Extract the last update time from feed."""
        try:
            # Try different fields for last update
            for field in ['updated_parsed', 'published_parsed']:
                if hasattr(feed, field) and getattr(feed, field):
                    return datetime(*getattr(feed, field)[:6])

            # Check entries for recent updates
            if feed.entries:
                for entry in feed.entries[:3]:
                    for field in ['updated_parsed', 'published_parsed']:
                        if hasattr(entry, field) and getattr(entry, field):
                            return datetime(*getattr(entry, field)[:6])

        except Exception:
            pass

        return None

    def _estimate_update_frequency(self, feed) -> str:
        """Estimate how frequently the feed is updated."""
        try:
            if not feed.entries:
                return "Unknown"

            # Check recent entries
            recent_entries = [e for e in feed.entries[:10]
                            if hasattr(e, 'updated_parsed') or hasattr(e, 'published_parsed')]

            if len(recent_entries) < 2:
                return "Unknown"

            # Calculate average time between entries
            times = []
            for entry in recent_entries:
                for field in ['updated_parsed', 'published_parsed']:
                    if hasattr(entry, field) and getattr(entry, field):
                        times.append(datetime(*getattr(entry, field)[:6]))
                        break

            if len(times) < 2:
                return "Unknown"

            times.sort(reverse=True)
            time_diffs = [(times[i] - times[i+1]).total_seconds() for i in range(len(times)-1)]
            avg_seconds = sum(time_diffs) / len(time_diffs)

            if avg_seconds < 86400:  # Less than 1 day
                return "Daily"
            elif avg_seconds < 604800:  # Less than 1 week
                return "Weekly"
            else:
                return "Monthly"

        except Exception:
            return "Unknown"

    def _check_topic_relevance(self, feed, topic: str) -> float:
        """
        Enhanced relevance scoring with fuzzy matching.

        Uses both exact matching, substring matching, and fuzzy word matching to catch
        related terminology without maintaining synonym mappings.
        """
        from difflib import SequenceMatcher

        topic_lower = topic.lower()
        score = 0.0

        # Check title and description
        title = feed.feed.get('title', '')
        description = feed.feed.get('description', '')

        title_lower = title.lower()
        description_lower = description.lower()

        # EXACT MATCHES (highest weight)
        # Title relevance
        if topic_lower in title_lower:
            score += 0.35

        # Description relevance
        if topic_lower in description_lower:
            score += 0.15

        # SUBSTRING MATCHING (catch partial matches like "security" in "cybersecurity")
        # Only apply if no exact match already found
        if topic_lower not in title_lower and topic_lower not in description_lower:
            for word in topic_lower.split():
                if len(word) >= 4:  # Only meaningful words
                    # Check if word appears as substring in title
                    if word in title_lower and word not in title_lower.split():
                        score += 0.08
                    # Check if word appears as substring in description
                    if word in description_lower and word not in description_lower.split():
                        score += 0.04

        # FUZZY WORD MATCHING (catch related terminology)
        def fuzzy_word_match(text: str, target: str) -> float:
            """Check if any words in target closely match words in text."""
            if not text or not target:
                return 0.0

            text_words = set(text.lower().split())
            target_words = set(target.lower().split())

            best_match = 0.0
            for t_word in target_words:
                for text_word in text_words:
                    # Calculate similarity
                    similarity = SequenceMatcher(None, t_word, text_word).ratio()
                    # Only count if similarity is high (0.7+)
                    if similarity >= 0.7:
                        best_match = max(best_match, similarity)

            return best_match

        # Fuzzy title match (if no exact match)
        if topic_lower not in title_lower:
            fuzzy_title = fuzzy_word_match(title, topic)
            score += fuzzy_title * 0.20  # Up to 0.20 for fuzzy title match

        # Fuzzy description match (if no exact match)
        if topic_lower not in description_lower:
            fuzzy_desc = fuzzy_word_match(description, topic)
            score += fuzzy_desc * 0.10  # Up to 0.10 for fuzzy description match
        
        # Check recent articles for topic relevance
        article_relevance = 0
        for entry in feed.entries[:10]:
            entry_title = entry.get('title', '')
            entry_summary = entry.get('summary', '')
            
            entry_title_lower = entry_title.lower()
            entry_summary_lower = entry_summary.lower()
            
            # Exact matches
            if topic_lower in entry_title_lower:
                article_relevance += 0.04
            if topic_lower in entry_summary_lower:
                article_relevance += 0.015
            
            # Fuzzy matches (only if no exact)
            if topic_lower not in entry_title_lower:
                article_relevance += fuzzy_word_match(entry_title, topic) * 0.03
            if topic_lower not in entry_summary_lower:
                article_relevance += fuzzy_word_match(entry_summary, topic) * 0.01

        score += min(article_relevance, 0.40)  # Cap article relevance at 0.40

        return min(score, 1.0)

    def _get_topic_specific_queries(self, topic: str) -> List[str]:
        """
        Generate domain-specific search queries using fuzzy variations.
        
        Auto-generates variations without maintaining hardcoded mappings.
        """
        topic_lower = topic.lower().strip()
        queries = []
        
        # Simple auto-generated variations
        # 1. Split compound words
        if len(topic_lower.split()) > 1:
            words = topic_lower.split()
            # Try first word only (e.g., "machine learning" -> "machine")
            queries.append(f'{words[0]} RSS feeds')
            # Try last word only (e.g., "machine learning" -> "learning")
            queries.append(f'{words[-1]} RSS feeds')
        
        # 2. Add common suffixes/prefixes
        queries.extend([
            f'{topic_lower} tech RSS feeds',
            f'{topic_lower} industry news feed',
            f'{topic_lower} technology blog',
        ])
        
        return queries

    def _generate_search_queries(self, topic: str) -> List[str]:
        """Generate enhanced RSS-specific search queries.
        
        Prioritizes topic-specific queries, then simpler patterns.
        More complex patterns are attempted as fallbacks.
        """
        # Shorten topic for better search results
        topic_short = self._shorten_topic(topic)
        
        # NEW: Get topic-specific queries first (highest priority)
        topic_specific = self._get_topic_specific_queries(topic_short)
        
        # BASE: Simple, effective queries (tested to work)
        simple_queries = [
            f'{topic_short} RSS feeds',
            f'best {topic_short} blogs',
            f'{topic_short} news feed',
            f'{topic_short} blog RSS',
            f'{topic_short} publications RSS'
        ]
        
        # FALLBACK 1: RSS-specific operators (used if simple queries fail)
        rss_specific = [
            f'"{topic_short}" RSS feed',
            f'{topic_short} RSS feed URL',
            f'"{topic_short}" "subscribe via RSS"',
        ]
        
        # FALLBACK 2: Platform-specific (Medium, Substack, etc.)
        platform_queries = [
            f'site:medium.com "{topic_short}"/rss',
            f'site:substack.com "{topic_short}"/feed',
            f'wordpress.com "{topic_short}"/feed',
        ]
        
        # FALLBACK 3: Feed aggregators (Feedspot, etc.)
        aggregator_queries = [
            f'site:feedspot.com "{topic_short}"',
            f'RSS "{topic_short}" site:feedspot.com',
        ]
        
        # Combine: prioritize topic-specific and simple queries first
        all_queries = topic_specific + simple_queries + rss_specific + platform_queries + aggregator_queries
        
        # Return more queries to increase success rate (increased from 8 to 10)
        return all_queries[:10]
    
    def _shorten_topic(self, topic: str) -> str:
        """Shorten topic names for better search results.
        
        Examples:
            "artificial intelligence" → "AI"
            "machine learning" → "machine learning" (kept full to avoid ambiguity)
            "blockchain technology" → "blockchain"
        """
        topic = topic.strip().lower()
        
        # ONLY abbreviate if unambiguous (avoid "ML", "DL" etc. that are too generic)
        abbreviations = {
            'artificial intelligence': 'AI',
            'natural language processing': 'NLP',
            'computer vision': 'CV',
            'virtual reality': 'VR',
            'augmented reality': 'AR',
            'internet of things': 'IoT',
            'generative ai': 'GenAI',
            'large language models': 'LLM',
        }
        
        # Check for exact matches
        if topic in abbreviations:
            return abbreviations[topic]
        
        # Remove common suffixes
        suffixes = [' technology', ' tech', ' and technology', ' computing']
        for suffix in suffixes:
            if topic.endswith(suffix):
                return topic[:-len(suffix)]
        
        # Return original if no shortening applies
        # (keeps "machine learning", "deep learning" etc. as full phrases)
        return topic

    def _unwrap_redirect_url(self, url: str) -> str:
        """
        Unwrap redirect URLs from search engines.
        
        DuckDuckGo returns URLs like: https://duckduckgo.com/l/?uddg=https://example.com
        This extracts the actual target URL.
        
        Args:
            url: Potentially wrapped URL
            
        Returns:
            Unwrapped URL, or original if no wrapping detected
        """
        if not url:
            return url
        
        # Check for DuckDuckGo redirect
        if 'duckduckgo.com/l/?uddg=' in url:
            try:
                # Extract the encoded URL
                parsed = urlparse(url)
                # Get the uddg parameter value
                if parsed.query:
                    for param in parsed.query.split('&'):
                        if param.startswith('uddg='):
                            encoded_url = param[5:]  # Remove 'uddg='
                            unwrapped = unquote(encoded_url)
                            logger.debug(f"Unwrapped DuckDuckGo URL: {unwrapped[:80]}")
                            return unwrapped
            except Exception as e:
                logger.debug(f"Failed to unwrap DuckDuckGo URL: {e}")
        
        return url

    def _extract_rss_urls(self, text: str) -> List[str]:
        """Extract RSS URLs from text."""
        if not text:
            return []

        urls = []

        # Simple regex patterns for RSS URLs
        patterns = [
            r'https?://[^\s\'"<>]+/feed[^\s]*',
            r'https?://[^\s\'"<>]+/rss[^\s]*',
            r'https?://[^\s\'"<>]+\.xml[^\s]*',
            r'https?://[^\s\'"<>]+/atom[^\s]*'
        ]

        for pattern in patterns:
            found_urls = re.findall(pattern, text, re.IGNORECASE)
            urls.extend(found_urls)

        # Look for RSS/Atom link tags
        link_pattern = r'<link[^>]*rel=["\'](?:alternate|service|feed)["\'][^>]*href=["\']([^"\'>]+)["\']'
        links = re.findall(link_pattern, text, re.IGNORECASE)
        urls.extend(links)

        return list(set(urls))  # Remove duplicates

    def _score_feed_quality(self, feed_data: Dict[str, Any]) -> float:
        """Score feed quality based on various factors."""
        score = 0.0

        # Base score for having data
        if feed_data:
            score += 0.3

        # Article count (more is better, up to 50)
        article_count = feed_data.get('article_count', 0)
        if article_count > 0:
            score += min(article_count / 50.0, 0.3)

        # Recency (recent articles are better)
        recent_count = feed_data.get('recent_article_count', 0)
        if recent_count > 0:
            score += min(recent_count / 10.0, 0.2)

        # Domain reputation
        feed_url = feed_data.get('url', '')
        url_lower = feed_url.lower()
        high_quality_domains = [
            'techcrunch.com', 'mit.edu', 'stanford.edu', 'nature.com', 'science.org',
            'wsj.com', 'bloomberg.com', 'reuters.com', 'forbes.com', 'hbr.org',
            'arxiv.org', 'ieee.org', 'acm.org', 'springer.com', 'wiley.com',
            'medium.com', 'substack.com', 'wordpress.com'
        ]

        for domain in high_quality_domains:
            if domain in url_lower:
                score += 0.2
                break

        return min(score, 1.0)

    def _get_feed_confidence_level(self, score: float) -> str:
        """Convert score to confidence level."""
        if score >= 0.7:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"


class FeedValidator:
    """Validate RSS feeds and assess quality metrics."""

    def __init__(self):
        self.discovery_engine = FeedDiscoveryEngine()

    def validate_feed(self, rss_url: str, topic: str) -> Optional[Dict[str, Any]]:
        """
        Validate RSS feed and extract metadata.

        Returns feed information if valid, None otherwise.
        """
        return self.discovery_engine._validate_feed(rss_url, topic)

    def get_feed_preview(self, rss_url: str, max_articles: int = 3) -> List[Dict[str, str]]:
        """Get preview articles from a feed."""
        try:
            headers = {'User-Agent': 'AI-News-Feed-Discovery/1.0'}
            response = requests.get(rss_url, headers=headers, timeout=10)
            response.raise_for_status()

            feed = feedparser.parse(response.content)

            articles = []
            for entry in feed.entries[:max_articles]:
                articles.append({
                    'title': entry.get('title', 'No title'),
                    'url': entry.get('link', ''),
                    'published': entry.get('published', 'Unknown date')
                })

            return articles

        except Exception as e:
            logger.error(f"Error getting feed preview from {rss_url}: {e}")
            return []
