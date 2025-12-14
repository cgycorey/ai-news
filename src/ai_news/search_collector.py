"""Search engine collector for AI news using web search APIs."""

import urllib.request
import urllib.parse
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any
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
            
            # Parse results from HTML
            results = []
            
            # Find result blocks
            result_pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?<a[^>]*class="result__a"[^>]*>(.*?)</a>'
            matches = re.findall(result_pattern, html_content, re.DOTALL)
            
            for i, (url, title, snippet) in enumerate(matches[:max_results]):
                # Clean up HTML entities and text
                title = clean_text_content(title)
                snippet = clean_text_content(snippet)
                
                # Clean URL
                url = html.unescape(url).strip()
                if url.startswith('//'):
                    url = 'https:' + url
                elif url.startswith('/l/?uddg='):
                    continue  # Skip DuckDuckGo redirect links
                
                if title and snippet and url and not url.startswith('http://duckduckgo.com'):
                    results.append({
                        'title': title,
                        'content': snippet,
                        'url': url,
                        'source': 'DuckDuckGo Search'
                    })
            
            return results
            
        except Exception as e:
            print(f"Error searching DuckDuckGo for '{query}': {e}")
            return []
    
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
                
                if title_elem is not None and link_elem is not None:
                    title = title_elem.text or ""
                    url = link_elem.text or ""
                    
                    # Clean description
                    if desc_elem is not None:
                        content = desc_elem.text or ""
                        content = clean_text_content(content)
                    else:
                        content = ""
                    
                    if title and url:
                        results.append({
                            'title': title,
                            'content': content,
                            'url': url,
                            'source': 'Bing News Search'
                        })
            
            return results
            
        except Exception as e:
            print(f"Error searching Bing News for '{query}': {e}")
            return []
    
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
            
            # Search DuckDuckGo
            ddg_results = self.search_duckduckgo(query, max_results=5)
            
            # Search Bing News
            bing_results = self.search_bing_news(query, max_results=5)
            
            # Combine and process results
            all_results = ddg_results + bing_results
            
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
                        article = Article(
                            title=title,
                            content=content,
                            summary=content[:200] + '...' if len(content) > 200 else content,
                            url=result['url'],
                            author='',
                            published_at=datetime.now(),
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