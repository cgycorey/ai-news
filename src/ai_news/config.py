"""Simple configuration management for AI News."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScheduleConfig:
    """Configuration for news collection scheduling."""
    enabled: bool = False
    interval: str = "daily"  # hourly, daily, weekly
    last_collection: Optional[str] = None
    next_collection: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduleConfig':
        return cls(
            enabled=data.get('enabled', False),
            interval=data.get('interval', 'daily'),
            last_collection=data.get('last_collection'),
            next_collection=data.get('next_collection')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'interval': self.interval,
            'last_collection': self.last_collection,
            'next_collection': self.next_collection
        }


class FeedConfig:
    """Configuration for a single RSS feed."""
    
    def __init__(self, name: str, url: str, category: str = "general", 
                 enabled: bool = True, ai_keywords: List[str] | None = None):
        # Validate inputs
        if not name or not name.strip():
            raise ValueError("Feed name cannot be empty")
        if not url or not url.strip():
            raise ValueError("Feed URL cannot be empty")
        if not url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL format: {url}")
        
        self.name = name.strip()
        self.url = url.strip()
        self.category = category.strip().lower() if category else "general"
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
        try:
            return cls(**data)
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid feed configuration: {e}")
            raise ValueError(f"Invalid feed configuration: {e}")


@dataclass
class Config:
    """Main application configuration."""
    database_path: str = "ai_news.db"
    feeds: List[FeedConfig] = field(default_factory=list)
    max_articles_per_feed: int = 50
    collection_interval_hours: int = 6
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    config_path: Optional[Path] = field(default=None, init=False)
    
    def __post_init__(self):
        # Validate inputs
        if self.max_articles_per_feed <= 0:
            raise ValueError("max_articles_per_feed must be positive")
        if self.collection_interval_hours <= 0:
            raise ValueError("collection_interval_hours must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        result = {
            "database_path": self.database_path,
            "feeds": [feed.to_dict() for feed in self.feeds],
            "max_articles_per_feed": self.max_articles_per_feed,
            "collection_interval_hours": self.collection_interval_hours,
        }
        if self.schedule:
            result['schedule'] = self.schedule.to_dict()
        return result

    @classmethod
    def load(cls, config_path: Path) -> 'Config':
        """Load configuration from JSON file."""
        if not config_path.exists():
            # Create default config
            default_config = cls._create_default()
            default_config.save(config_path)
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            logger.info("Creating default configuration instead")
            default_config = cls._create_default()
            default_config.save(config_path)
            return default_config
        
        schedule_data = data.get('schedule', {})
        schedule = ScheduleConfig.from_dict(schedule_data)
        
        # Rest of existing from_dict logic...
        config = cls(
            database_path=data.get('database_path', 'ai_news.db'),
            feeds=[FeedConfig.from_dict(feed_data) for feed_data in data.get("feeds", [])],
            max_articles_per_feed=data.get('max_articles_per_feed', 50),
            collection_interval_hours=data.get('collection_interval_hours', 6),
            schedule=schedule
        )
        config.config_path = config_path
        return config

    def save(self, config_path: Optional[Path] = None):
        """Save configuration to JSON file."""
        path = config_path or self.config_path
        if not path:
            raise ValueError("No config path specified")
        
        config_data = self.to_dict()
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)

    @staticmethod
    def _create_default() -> 'Config':
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
        ]
        
        return Config(feeds=default_feeds)