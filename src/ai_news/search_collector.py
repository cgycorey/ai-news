"""Search engine collector for AI news using web search APIs."""

import urllib.request
import urllib.parse
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import html
import time

from .config import FeedConfig
from .database import Article, Database
from .security_utils import (
    parse_xml_safe, clean_text_content, validate_url, safe_urlopen
)


class SearchEngineCollector:
    """Collect articles from search engines for AI + topic queries."""
    
    def __init__(self, database: Database):
        self.database = database
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; AI-News-Collector/1.0)'
        }
        # Track page fetches to avoid excessive requests
        self._page_fetch_count = 0
        self._max_page_fetches = 5  # Limit per search_topic call
    
    def search_duckduckgo(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo's HTML version (no API key needed)."""
        try:
            # DuckDuckGo instant answer API (HTML format)
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"
            
            # Validate URL
            is_valid, error = validate_url(url)
            if not is_valid:
                print(f"DuckDuckGo URL validation failed: {error}")
                return []
            
            # Use safe URL opener
            response = safe_urlopen(url, headers=self.headers, timeout=30)
            if response is None:
                return []
            
            with response:
                html_content = response.read().decode('utf-8', errors='ignore')
            
            # Parse results from HTML using BeautifulSoup
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            results = []
            
            # Find all result blocks
            result_blocks = soup.find_all('div', class_='result')
            
            for result in result_blocks[:max_results]:
                # Extract title
                title_elem = result.find('a', class_='result__a')
                title = title_elem.get_text().strip() if title_elem else ''
                
                # Extract URL
                url_elem = result.find('a', class_='result__a')
                url = url_elem.get('href') if url_elem else ''
                
                # Extract snippet
                snippet_elem = result.find('a', class_='result__snippet')
                snippet = snippet_elem.get_text().strip() if snippet_elem else ''
                
                # Extract date from DuckDuckGo result (if available)
                # DDG shows dates in: <span>YYYY-MM-DD</span> within result__extras__url
                published_date = None
                extras_div = result.find('div', class_='result__extras__url')
                if extras_div:
                    # Look for span with date pattern
                    date_span = extras_div.find('span')
                    if date_span:
                        date_text = date_span.get_text().strip()
                        # Parse YYYY-MM-DD format
                        try:
                            published_date = datetime.strptime(date_text, '%Y-%m-%d')
                        except ValueError:
                            pass

                # Clean URL
                url = html.unescape(url).strip()
                if url.startswith('//'):
                    url = 'https:' + url
                elif url.startswith('/l/?uddg='):
                    continue

                # Extract real URL from DuckDuckGo redirect
                if 'duckduckgo.com/l/?uddg=' in url:
                    parsed = urllib.parse.urlparse(url)
                    params = urllib.parse.parse_qs(parsed.query)
                    if 'uddg' in params:
                        url = urllib.parse.unquote(params['uddg'][0])
                    else:
                        continue

                if title and snippet and url and not url.startswith('http://duckduckgo.com') and not url.startswith('https://duckduckgo.com'):
                    results.append({
                        'title': title,
                        'content': snippet,
                        'url': url,
                        'source': 'DuckDuckGo Search',
                        'published_date': published_date
                    })
            
            return results
            
        except Exception as e:
            print(f"Error searching DuckDuckGo for '{query}': {e}")
            return []
    
    def search_searxng(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search using SearXNG instance."""
        results = []
        
        try:
            # SearXNG instance URL
            url = "https://cgycorey-searxng3.hf.space/search"
            
            # Prepare search parameters
            params = {
                'q': query,
                'format': 'json',
                'engines': 'google,bing,duckduckgo',
                'language': 'en',
                'time_range': None,
                'safesearch': 1
            }
            
            # Validate URL
            search_url = f"{url}?{urllib.parse.urlencode(params)}"
            is_valid, error = validate_url(search_url)
            if not is_valid:
                print(f"SearXNG URL validation failed: {error}")
                return []
            
            # Use safe URL opener
            response = safe_urlopen(search_url, headers=self.headers, timeout=30)
            if response is None:
                return []
            
            with response:
                json_content = response.read().decode('utf-8', errors='ignore')
            
            # Parse JSON response
            search_data = json.loads(json_content)
            
            # Extract results from JSON
            for result in search_data.get('results', [])[:max_results]:
                # Get engine name for source
                engine = result.get('engine', ['SearXNG'])
                engine_name = engine[0] if isinstance(engine, list) and engine else 'SearXNG'
                
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'content': result.get('content', ''),
                    'source': f'{engine_name} (SearXNG)',
                    'engine': result.get('engine', []),
                    'score': result.get('score', 0),
                    'category': result.get('category', '')
                })
            
        except Exception as e:
            print(f"SearXNG search error for '{query}': {e}")
            return []
        
        return results
    
    def search_bing_news(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search using Bing News (no API key needed)."""
        try:
            url = f"https://www.bing.com/news/search?q={urllib.parse.quote_plus(query)}&format=rss"
            
            # Validate URL
            is_valid, error = validate_url(url)
            if not is_valid:
                print(f"Bing News URL validation failed: {error}")
                return []
            
            # Use safe URL opener
            response = safe_urlopen(url, headers=self.headers, timeout=30)
            if response is None:
                return []
            
            with response:
                content = response.read().decode('utf-8', errors='ignore')
            
            # Parse RSS securely from search results
            root = parse_xml_safe(content)
            
            results = []
            
            # Find items in RSS
            for item in root.findall('.//item')[:max_results]:
                title_elem = item.find('title')
                link_elem = item.find('link')
                desc_elem = item.find('description')
                pubdate_elem = item.find('pubDate')

                if title_elem is not None and link_elem is not None:
                    title = title_elem.text or ""
                    url = link_elem.text or ""

                    # Clean description
                    if desc_elem is not None:
                        content = desc_elem.text or ""
                        content = clean_text_content(content)
                    else:
                        content = ""

                    # Parse pubDate if available
                    published_date = None
                    if pubdate_elem is not None and pubdate_elem.text:
                        published_date = self.parse_rss_date(pubdate_elem.text)

                    if title and url:
                        results.append({
                            'title': title,
                            'content': content,
                            'url': url,
                            'source': 'Bing News Search',
                            'published_date': published_date
                        })
            
            return results
            
        except Exception as e:
            print(f"Error searching Bing News for '{query}': {e}")
            return []

    def parse_rss_date(self, date_str: str) -> Optional[datetime]:
        """Parse RSS date string (RFC 822/2822 format) to datetime."""
        if not date_str:
            return None

        # Common RSS date formats
        formats = [
            '%a, %d %b %Y %H:%M:%S %Z',  # RFC 822
            '%a, %d %b %Y %H:%M:%S %z',  # RFC 822 with timezone
            '%a, %d %b %Y %H:%M:%S',     # RFC 822 without timezone
            '%Y-%m-%dT%H:%M:%S%z',       # ISO 8601
            '%Y-%m-%dT%H:%M:%SZ',        # ISO 8601 UTC
            '%Y-%m-%d %H:%M:%S',         # Simple format
            '%Y-%m-%d',                  # Date only
        ]

        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str.strip(), fmt)
                # Validate: date shouldn't be in the future
                if parsed_date > datetime.now():
                    return None
                return parsed_date
            except ValueError:
                continue

        return None

    def clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        if not content:
            return ""
        
        # Use secure HTML sanitization
        content = clean_text_content(content)
        
        # Limit length
        if len(content) > 1000:
            content = content[:1000].rsplit(' ', 1)[0] + '...'
        
        return content
    
    def extract_date_from_url(self, url: str) -> Optional[datetime]:
        """Extract publish date from URL structure.

        Supports multiple URL date patterns:
        - /YYYY/MM/DD/ or /YYYY/MM/DD (most reliable)
        - /YYYYMMDDID (BusinessWire style: /home/20250303149241/)
        - YYYYMMDD at end of path (TechTimes style: /20241011/)
        - /YYYY/MM/ (set day to 1)
        """
        import re
        from datetime import datetime

        if not url:
            return None

        # Pattern 1: YYYY/MM/DD in URL (most reliable)
        match = re.search(r'/(\d{4})/(\d{1,2})/(\d{1,2})(?:/|$)', url)
        if match:
            try:
                year, month, day = match.groups()
                date = datetime(int(year), int(month), int(day))
                if date <= datetime.now():
                    return date
            except (ValueError, TypeError):
                pass

        # Pattern 2: YYYYMMDD format (BusinessWire: /home/20250303149241/)
        # Captures 8 digits followed by another digit (article ID)
        match = re.search(r'/(\d{4})(\d{2})(\d{2})\d', url)
        if match:
            try:
                year, month, day = match.groups()
                date = datetime(int(year), int(month), int(day))
                if date <= datetime.now():
                    return date
            except (ValueError, TypeError):
                pass

        # Pattern 3: YYYYMMDD at end of path or before extension (TechTimes: /20241011/)
        match = re.search(r'/(\d{4})(\d{2})(\d{2})(?:/|\.|$)', url)
        if match:
            try:
                year, month, day = match.groups()
                date = datetime(int(year), int(month), int(day))
                if date <= datetime.now():
                    return date
            except (ValueError, TypeError):
                pass

        # Pattern 4: YYYY/MM in URL (set day to 1) - check last to avoid false matches
        # Only match if YYYY/MM/DD pattern didn't match
        if not re.search(r'/(\d{4})/(\d{1,2})/(\d{1,2})', url):
            match = re.search(r'/(\d{4})/(\d{1,2})(?:/|$)', url)
            if match:
                try:
                    year, month = match.groups()
                    date = datetime(int(year), int(month), 1)
                    if date <= datetime.now():
                        return date
                except (ValueError, TypeError):
                    pass

        return None
    
    def fetch_article_date(self, url: str) -> Optional[datetime]:
        """Fetch article page and extract publication date from meta tags."""
        import re
        from datetime import datetime
        
        if not url:
            return None
        
        try:
            # Validate URL
            is_valid, error = validate_url(url)
            if not is_valid:
                return None
            
            # Fetch page with timeout
            response = safe_urlopen(url, headers=self.headers, timeout=10)
            if response is None:
                return None
            
            with response:
                html_content = response.read().decode('utf-8', errors='ignore')
            
            # Common date patterns in meta tags (in order of reliability)
            patterns = [
                # article:published_time (most reliable)
                (r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']', '%Y-%m-%dT%H:%M:%S'),
                # date meta tag
                (r'<meta[^>]+name=["\']date["\'][^>]+content=["\']([^"\']+)["\']', '%Y-%m-%d'),
                # pubdate
                (r'<meta[^>]+name=["\']pubdate["\'][^>]+content=["\']([^"\']+)["\']', '%Y-%m-%d'),
                # DC.date
                (r'<meta[^>]+name=["\']DC.date["\'][^>]+content=["\']([^"\']+)["\']', '%Y-%m-%d'),
                # time datetime attribute
                (r'<time[^>]+datetime=["\']([^"\']+)["\']', '%Y-%m-%dT%H:%M:%S'),
            ]
            
            for pattern, date_format in patterns:
                match = re.search(pattern, html_content, re.IGNORECASE)
                if match:
                    date_str = match.group(1)
                    try:
                        # Try parsing with the specified format
                        if 'T' in date_str:
                            # ISO 8601 format
                            parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        else:
                            parsed_date = datetime.strptime(date_str[:19], date_format)
                        return parsed_date
                    except (ValueError, TypeError):
                        # Try different formats
                        try:
                            parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            return parsed_date
                        except:
                            continue
            
        except Exception as e:
            # Silently fail - date extraction is best-effort
            pass
        
        return None
    
    def extract_publish_date(self, content: str, url: str = '') -> Optional[datetime]:
        """Extract publish date from content snippet or URL."""
        import re
        from datetime import datetime
        
        # Method 1: Extract from URL (fastest, most reliable for many sites)
        if url:
            url_date = self.extract_date_from_url(url)
            if url_date:
                return url_date
        
        # Method 2: Extract from content snippet (SearXNG)
        if content:
            # Format: "Aug 13, 2025 ·", "Jan 7, 2020 ·", etc.
            date_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}'
            
            match = re.search(date_pattern, content)
            if match:
                try:
                    date_str = match.group(0)
                    parsed_date = datetime.strptime(date_str, '%b %d, %Y')
                    return parsed_date
                except ValueError:
                    pass
        
        # Method 3: Fetch article page for DuckDuckGo/Bing results (slower)
        # Only do this if we couldn't find date elsewhere
        # This is expensive, so we'll do it selectively in the calling code
        return None
    
    def is_ai_relevant(self, title: str, content: str) -> tuple[bool, List[str]]:
        """Check if content is AI-relevant."""
        ai_keywords = [
            "artificial intelligence", "machine learning", "deep learning", "neural network",
            "AI", "LLM", "GPT", "ChatGPT", "OpenAI", "Anthropic", "Claude",
            "algorithm", "automation", "predictive", "data science", "analytics"
        ]
        
        text = (title + " " + content).lower()
        found_keywords = [kw for kw in ai_keywords if kw.lower() in text]
        
        return len(found_keywords) > 0, found_keywords
    
    def search_topic(self, topic: str, days_back: int = 7, max_results: int = 15) -> List[Article]:
        """Search for AI + topic articles."""
        articles = []
        
        # Generate search queries
        queries = [
            f"AI {topic}",
            f"artificial intelligence {topic}",
            f"machine learning {topic}",
            f"{topic} technology",
            f"{topic} automation"
        ]
        
        for query in queries:
            print(f"  Searching: {query}")
            
            # Search SearXNG (has dates in snippets)
            searxng_results = self.search_searxng(query, max_results=5)
            
            # Search DuckDuckGo
            ddg_results = self.search_duckduckgo(query, max_results=3)
            
            # Search Bing News
            bing_results = self.search_bing_news(query, max_results=2)
            
            # Combine and process results
            all_results = searxng_results + ddg_results + bing_results
            
            for result in all_results:
                try:
                    # Clean title and content
                    title = self.clean_content(result['title'])
                    content = self.clean_content(result['content'])
                    
                    if not title or not content:
                        continue
                    
                    # Check AI relevance
                    is_ai, keywords = self.is_ai_relevant(title, content)
                    
                    if is_ai:
                        # Step 1: Use RSS date if available from Bing News
                        published_at = result.get('published_date')

                        # Step 2: Extract from URL (fast, many sites)
                        if not published_at:
                            published_at = self.extract_date_from_url(result['url'])

                        # Step 3: Extract from SearXNG content snippets
                        if not published_at and content:
                            # Format: "Aug 13, 2025 ·"
                            date_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}'
                            match = re.search(date_pattern, content)
                            if match:
                                try:
                                    date_str = match.group(0)
                                    published_at = datetime.strptime(date_str, '%b %d, %Y')
                                except ValueError:
                                    pass

                        # Step 4: For DuckDuckGo results ONLY (Bing now has RSS dates)
                        # Fetch article page as last resort (expensive, rate-limited)
                        if not published_at and 'DuckDuckGo' in result.get('source', ''):
                            if self._page_fetch_count < self._max_page_fetches:
                                published_at = self.fetch_article_date(result['url'])
                                self._page_fetch_count += 1

                        # Note: published_at can be None - digest will show "Unknown"
                        
                        article = Article(
                            title=title,
                            content=content,
                            summary=content[:200] + '...' if len(content) > 200 else content,
                            url=result['url'],
                            author='',
                            published_at=published_at,
                            source_name=result['source'],
                            category='search',
                            ai_relevant=is_ai,
                            ai_keywords_found=keywords
                        )
                        
                        articles.append(article)
                        
                except Exception as e:
                    print(f"    Error processing result: {e}")
                    continue
            
            # Small delay between searches
            time.sleep(1)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        for article in articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        return unique_articles
    
    def collect_trending_topics(self) -> List[Article]:
        """Collect articles for trending AI topics."""
        trending_topics = [
            "healthcare", "insurance", "finance", "banking", "manufacturing",
            "retail", "transportation", "education", "agriculture", "energy",
            "cybersecurity", "robotics", "autonomous", "drug discovery",
            "customer service", "supply chain", "compliance"
        ]
        
        all_articles = []
        
        for topic in trending_topics:
            print(f"Collecting AI articles for: {topic}")
            articles = self.search_topic(topic, days_back=7, max_results=10)
            
            added_count = 0
            for article in articles:
                if self.database.add_article(article):
                    added_count += 1
                    all_articles.append(article)
            
            print(f"  Added {added_count}/{len(articles)} articles")
            
            # Be respectful with search rate limiting
            time.sleep(2)
        
        return all_articles