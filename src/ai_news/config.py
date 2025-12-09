"""Simple configuration management for AI News."""

from pathlib import Path
from typing import List, Dict, Any
import json


class FeedConfig:
    """Configuration for a single RSS feed."""
    
    def __init__(self, name: str, url: str, category: str = "general", 
                 enabled: bool = True, ai_keywords: List[str] | None = None):
        self.name = name
        self.url = url
        self.category = category
        self.enabled = enabled
        self.ai_keywords = ai_keywords or [
            "artificial intelligence", "machine learning", "deep learning",
            "neural network", "LLM", "GPT", "ChatGPT", "OpenAI", "Anthropic",
            "AI", "ML", "AGI", "transformer", "BERT", "NLP", "computer vision"
        ]
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "category": self.category,
            "enabled": self.enabled,
            "ai_keywords": self.ai_keywords
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FeedConfig":
        return cls(**data)


class Config:
    """Main application configuration."""
    
    def __init__(self, database_path: str = "ai_news.db", feeds: List[FeedConfig] | None = None,
                 max_articles_per_feed: int = 50, collection_interval_hours: int = 6):
        self.database_path = database_path
        self.feeds = feeds or []
        self.max_articles_per_feed = max_articles_per_feed
        self.collection_interval_hours = collection_interval_hours

    @classmethod
    def load(cls, config_path: Path) -> "Config":
        """Load configuration from JSON file."""
        if not config_path.exists():
            # Create default config
            default_config = cls._create_default()
            default_config.save(config_path)
            return default_config
        
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        feeds = [FeedConfig.from_dict(feed_data) for feed_data in data.get("feeds", [])]
        
        return cls(
            database_path=data.get("database_path", "ai_news.db"),
            feeds=feeds,
            max_articles_per_feed=data.get("max_articles_per_feed", 50),
            collection_interval_hours=data.get("collection_interval_hours", 6)
        )

    def save(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        data = {
            "database_path": self.database_path,
            "feeds": [feed.to_dict() for feed in self.feeds],
            "max_articles_per_feed": self.max_articles_per_feed,
            "collection_interval_hours": self.collection_interval_hours
        }
        
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _create_default() -> "Config":
        """Create default configuration with sample feeds."""
        default_feeds = [
            FeedConfig(
                name="TechCrunch AI",
                url="https://techcrunch.com/category/artificial-intelligence/feed/",
                category="tech"
            ),
            FeedConfig(
                name="MIT Technology Review AI",
                url="https://www.technologyreview.com/topic/artificial-intelligence/feed/",
                category="research"
            ),
            FeedConfig(
                name="Ars Technica AI",
                url="https://arstechnica.com/tag/artificial-intelligence/feed/",
                category="tech"
            ),
            FeedConfig(
                name="Towards Data Science",
                url="https://towardsdatascience.com/feed",
                category="tutorial"
            ),
            FeedConfig(
                name="arXiv Computer Science - Machine Learning",
                url="http://rss.arxiv.org/rss/cs.LG",
                category="research",
                ai_keywords=["machine learning", "neural network", "deep learning", "algorithm"]
            ),
            FeedConfig(
                name="Hacker News",
                url="https://hnrss.org/frontpage",
                category="general",
                ai_keywords=["AI", "artificial intelligence", "machine learning", "OpenAI", "GPT"]
            ),
            
            # MAJOR NEWS SOURCES with better AI coverage
            FeedConfig(
                name="Reuters AI News",
                url="https://www.reuters.com/archis/rss/ai-automation-intelligence",
                category="tech",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "automation", "algorithms", "deep learning", "neural networks"]
            ),
            
            FeedConfig(
                name="Bloomberg Technology",
                url="https://feeds.bloomberg.com/technology/news.rss",
                category="tech",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "fintech", "automation", "software", "algorithms"]
            ),
            
            FeedConfig(
                name="CNBC Technology",
                url="https://www.cnbc.com/id/19832330/device/rss/rss.html",
                category="tech",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "tech", "automation", "software", "startups"]
            ),
            
            FeedConfig(
                name="BBC Technology",
                url="https://feeds.bbci.co.uk/news/technology/rss.xml",
                category="tech",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "technology", "automation", "software"]
            ),
            
            FeedConfig(
                name="The Verge - AI",
                url="https://www.theverge.com/ai-artificial-intelligence/rss.xml",
                category="tech",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "LLM", "ChatGPT", "OpenAI", "Anthropic", "tech"]
            ),
            
            # Healthcare AI Sources (working feeds)
            FeedConfig(
                name="Fierce Healthcare",
                url="https://www.fiercehealthcare.com/rss/xml",
                category="healthcare",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "healthcare", "medical", "digital health", "diagnosis", "treatment"]
            ),
            
            FeedConfig(
                name="MobiHealthNews",
                url="https://www.mobihealthnews.com/rss/feed",
                category="healthcare",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "healthcare", "digital health", "medical technology"]
            ),
            
            # Finance & FinTech AI Sources
            FeedConfig(
                name="FinTech Futures",
                url="https://www.finextra.com/press-releases/rss",
                category="finance",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "fintech", "banking", "insurance", "automation", "algorithms"]
            ),
            
            FeedConfig(
                name="American Banker",
                url="https://www.americanbanker.com/rss/news",
                category="finance",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "fintech", "banking", "automation", "algorithms"]
            ),
            
            # Insurance Industry News
            FeedConfig(
                name="Insurance Journal",
                url="https://www.insurancejournal.com/feed/",
                category="insurance",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "insurtech", "algorithmic", "automation", "predictive analytics", "data science", "deep learning", "neural network", "digital transformation", "claims automation", "underwriting AI"]
            ),
            
            FeedConfig(
                name="Insurance Business America",
                url="https://www.insurancebusinessmag.com/us/feed/",
                category="insurance",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "insurtech", "algorithmic", "automation", "predictive analytics", "claims automation", "underwriting AI", "digital insurance"]
            ),
            
            # General Tech and Business News
            FeedConfig(
                name="Wired",
                url="https://www.wired.com/feed/rss",
                category="tech",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "technology", "automation", "algorithm"]
            ),
            
            FeedConfig(
                name="The Verge",
                url="https://www.theverge.com/rss/index.xml",
                category="tech",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "tech", "technology", "automation"]
            ),
            
            FeedConfig(
                name="Fast Company",
                url="https://www.fastcompany.com/rss",
                category="business",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "business", "innovation", "automation"]
            ),
            
            # AI Research and Industry Sources
            FeedConfig(
                name="AI News",
                url="https://artificialintelligence-news.com/feed/",
                category="tech",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "LLM", "ChatGPT", "OpenAI", "Anthropic", "deep learning"]
            ),
            
            FeedConfig(
                name="VentureBeat AI",
                url="https://venturebeat.com/ai/feed/",
                category="tech",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "LLM", "ChatGPT", "OpenAI", "startups", "venture capital"]
            ),
            
            FeedConfig(
                name="The Register - AI",
                url="https://www.theregister.com/artificial_intelligence/headlines.atom",
                category="tech",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "LLM", "automation", "enterprise tech"]
            ),
            
            FeedConfig(
                name="InfoWorld",
                url="https://www.infoworld.com/category/artificial-intelligence/rss.xml",
                category="tech",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "enterprise", "automation", "software"]
            ),
            
            FeedConfig(
                name="KDnuggets",
                url="https://www.kdnuggets.com/feed/",
                category="research",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "data science", "analytics", "algorithms", "research"]
            ),
            
            FeedConfig(
                name="DeepMind Blog",
                url="https://deepmind.google/blog/feed/",
                category="research",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "deep learning", "neural networks", "research"]
            ),
            
            FeedConfig(
                name="OpenAI Blog",
                url="https://openai.com/blog/rss.xml",
                category="research",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "GPT", "LLM", "ChatGPT", "research"]
            ),
            
            FeedConfig(
                name="Anthropic Blog",
                url="https://www.anthropic.com/news/rss",
                category="research",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "Claude", "LLM", "research", "safety"]
            ),
            
            FeedConfig(
                name="Google AI Blog",
                url="https://ai.googleblog.com/feeds/posts/default",
                category="research",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "deep learning", "research", "Google", "TensorFlow"]
            ),
            
            FeedConfig(
                name="Microsoft Research",
                url="https://www.microsoft.com/en-us/research/blog/feed/",
                category="research",
                ai_keywords=["artificial intelligence", "AI", "machine learning", "research", "Microsoft", "Azure", "enterprise"]
            ),
        ]
        
        return Config(feeds=default_feeds)